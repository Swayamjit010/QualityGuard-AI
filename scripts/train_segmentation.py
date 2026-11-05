"""Train + evaluate the supervised defect-segmentation model (ResNet-18 U-Net) on KolektorSDD2.

    python scripts/train_segmentation.py [--epochs 14] [--neg-per-pos 3]

Protocol: fit on manifest 'train'; choose the epoch (early stopping on val Dice + val image AUROC) and the
image-level decision threshold on 'val'; report once on 'test' (the official KSDD2 test split).
Each epoch uses all defective training images plus a fresh random sample of ``neg_per_pos`` x as many
defect-free images (class imbalance ~1:8 otherwise dominates training on CPU).

Output: artifacts/checkpoints/ksdd2-segmentation-unet_resnet18-v1/
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

from ml.config import CHECKPOINT_DIR, VIS_DIR, absp, seed_everything  # noqa: E402
from ml.datasets.analysis import read_image  # noqa: E402
from ml.datasets.splitting import load_manifest  # noqa: E402
from ml.evaluation import metrics as M  # noqa: E402
from ml.segmentation.unet import ResNetUNet, image_score, seg_loss  # noqa: E402

RUN_ID = "ksdd2-segmentation-unet_resnet18-v1"
H, W = 384, 128
MEAN, STD = np.array([0.485, 0.456, 0.406], np.float32), np.array([0.229, 0.224, 0.225], np.float32)
SEG_PP = {"height": H, "width": W, "mean": MEAN.tolist(), "std": STD.tolist(), "color_mode": "rgb",
          "resize": "cv2.INTER_AREA to (W,H); masks INTER_NEAREST"}


def load_pair(r: dict):
    img = cv2.cvtColor(read_image(absp(r["path"])), cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
    if r.get("mask_path"):
        m = cv2.imread(str(absp(r["mask_path"])), cv2.IMREAD_GRAYSCALE)
        m = (cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST) > 127).astype(np.float32)
    else:
        m = np.zeros((H, W), np.float32)
    return img, m


def to_t(img: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(((img.astype(np.float32) / 255 - MEAN) / STD).transpose(2, 0, 1).copy())


class SegDS(Dataset):
    def __init__(self, rows, train: bool):
        self.rows, self.train = rows, train
        self.cache = {}

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        if r["path"] not in self.cache:
            self.cache[r["path"]] = load_pair(r)
        img, m = self.cache[r["path"]]
        if self.train:
            if np.random.rand() < 0.5:
                img, m = img[:, ::-1], m[:, ::-1]
            if np.random.rand() < 0.5:
                img, m = img[::-1], m[::-1]
            a = 1 + np.random.uniform(-0.15, 0.15); b = np.random.uniform(-15, 15)
            img = np.clip(img.astype(np.float32) * a + b, 0, 255).astype(np.uint8)
        return to_t(img), torch.from_numpy(np.ascontiguousarray(m))[None]


@torch.no_grad()
def run(model, rows, bs=16):
    model.eval()
    scores, maps = [], []
    for x, _ in DataLoader(SegDS(rows, False), batch_size=bs):
        p = torch.sigmoid(model(x))
        scores.append(image_score(p).numpy())
        maps.append(p[:, 0].numpy().astype(np.float16))
    return np.concatenate(scores), np.concatenate(maps)


def masks_of(rows):
    return np.stack([load_pair(r)[1] > 0.5 for r in rows])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=14)
    ap.add_argument("--neg-per-pos", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--finalize-only", action="store_true",
                    help="skip training: load best_state.pt saved by an interrupted run, rebuild the history from its "
                         "log, choose thresholds on val and evaluate once on test")
    ap.add_argument("--log", default="artifacts/logs/ksdd2_segmentation.log")
    a = ap.parse_args()
    seed_everything(42)
    torch.set_num_threads(max(2, torch.get_num_threads()))
    man = load_manifest("ksdd2")
    rows = {s: [r for r in man["samples"] if r["split"] == s] for s in ("train", "val", "test")}
    pos = [r for r in rows["train"] if r["label"] != "good"]
    neg = [r for r in rows["train"] if r["label"] == "good"]
    out = CHECKPOINT_DIR / RUN_ID
    out.mkdir(parents=True, exist_ok=True)
    (out / "status.json").write_text(json.dumps({"state": "running", "run_id": RUN_ID}))
    model = ResNetUNet(pretrained=True)
    enc = set(map(id, model.encoder_params()))
    opt = torch.optim.AdamW([{"params": list(model.encoder_params()), "lr": a.lr / 3},
                             {"params": [p for p in model.parameters() if id(p) not in enc], "lr": a.lr}],
                            weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    yv = np.array([int(r["label"] != "good") for r in rows["val"]])
    mv = masks_of(rows["val"])
    rng = np.random.default_rng(42)
    hist, best, best_ep, bad, t0 = [], -1.0, 0, 0, time.time()
    if a.finalize_only:
        import re
        for line in Path(a.log).read_text(encoding="utf-8").splitlines():
            m = re.match(r"\[" + re.escape(RUN_ID) + r"\] (\{.*\})$", line.strip())
            if m:
                hist.append(json.loads(m.group(1)))
        best_ep = max(hist, key=lambda h: h["select"])["epoch"]
        print(f"[{RUN_ID}] finalize-only: {len(hist)} logged epochs, best epoch {best_ep}", flush=True)
    for ep in range(1, 0 if a.finalize_only else a.epochs + 1):
        model.train()
        sample = pos + [neg[i] for i in rng.choice(len(neg), min(len(neg), a.neg_per_pos * len(pos)), replace=False)]
        dl = DataLoader(SegDS(sample, True), batch_size=a.batch_size, shuffle=True)
        tl, te = 0.0, time.time()
        for x, m in dl:
            opt.zero_grad()
            loss = seg_loss(model(x), m)
            loss.backward()
            opt.step()
            tl += loss.item() * len(x)
        sched.step()
        sv, pv = run(model, rows["val"])
        pm = M.pixel_metrics(mv, pv.astype(np.float32), 0.5)
        auroc = M.binary_anomaly_report(yv, sv)["image_auroc"]
        sel = 0.5 * (pm["mean_dice_defective"] or 0) + 0.5 * auroc
        h = {"epoch": ep, "train_loss": tl / len(sample), "val_dice": pm["mean_dice_defective"],
             "val_pixel_auroc": pm["pixel_auroc"], "val_image_auroc": auroc, "select": sel,
             "epoch_s": time.time() - te}
        hist.append(h)
        print(f"[{RUN_ID}] {json.dumps(h)}", flush=True)
        if sel > best:
            best, best_ep, bad = sel, ep, 0
            torch.save(model.state_dict(), out / "best_state.pt")
        else:
            bad += 1
            if bad >= a.patience:
                print(f"[{RUN_ID}] early stopping at epoch {ep}", flush=True)
                break
    model.load_state_dict(torch.load(out / "best_state.pt"))
    sv, pv = run(model, rows["val"])
    t_f1 = M.threshold_for_f1(yv, sv)
    t_rec = M.threshold_for_recall(yv, sv, 0.90)
    st, pt = run(model, rows["test"])
    yt = np.array([int(r["label"] != "good") for r in rows["test"]])
    mt = masks_of(rows["test"])
    test_pix = M.pixel_metrics(mt, pt.astype(np.float32), 0.5)
    # latency, single image
    x1 = SegDS(rows["test"][:1], False)[0][0][None]
    model.eval()
    with torch.no_grad():
        model(x1)
        ts = []
        for _ in range(10):
            t = time.perf_counter(); model(x1); ts.append(time.perf_counter() - t)
    res = {"run_id": RUN_ID, "task": "segmentation", "dataset": "ksdd2", "dataset_version": man["version"],
           "architecture": "U-Net decoder on ImageNet ResNet-18 encoder", "input_hw": [H, W],
           "split_sizes": {k: len(v) for k, v in rows.items()},
           "train_defective_images": len(pos), "train_defect_free_images": len(neg),
           "best_epoch": best_ep, "epochs_run": len(hist), "history": hist,
           "stopped_early_by_operator": bool(a.finalize_only),
           "image_threshold_f1_val": t_f1, "image_threshold_recall90_val": t_rec,
           "val": M.binary_anomaly_report(yv, sv, t_f1),
           "test": M.binary_anomaly_report(yt, st, t_f1),
           "test_at_recall90_threshold": M.binary_anomaly_report(yt, st, t_rec),
           "test_pixel": test_pix, "latency_ms_median": float(np.median(ts) * 1000),
           "train_time_s": time.time() - t0, "completed_at": datetime.now(timezone.utc).isoformat(),
           "image_score": "max of 5x5 average-pooled defect-probability map",
           "selection": "epoch maximising 0.5*val Dice + 0.5*val image AUROC; thresholds chosen on val"
                        + ("; training was stopped after the logged epochs to save time (max 14 planned)"
                           if a.finalize_only else "")}
    torch.save({"arch": "unet_resnet18", "state_dict": model.state_dict(), "preprocess": SEG_PP,
                "image_threshold": t_f1, "pixel_threshold": 0.5, "run_id": RUN_ID,
                "dataset_version": man["version"]}, out / "segmentation.pt")
    (out / "best_state.pt").unlink(missing_ok=True)
    M.save_json(res, out / "metrics.json")
    M.plot_history([{"epoch": h["epoch"], "train_loss": h["train_loss"], "val_loss": 1 - h["select"],
                     "train_acc": h["val_image_auroc"], "val_acc": h["val_dice"] or 0} for h in hist], out / "history.png")
    (out / "status.json").write_text(json.dumps({"state": "completed", "run_id": RUN_ID}))
    # example visualisations (held-out test)
    vis = VIS_DIR / RUN_ID
    vis.mkdir(parents=True, exist_ok=True)
    idx = [i for i in range(len(yt)) if yt[i]][:4] + [i for i in range(len(yt)) if not yt[i]][:2]
    for i in idx:
        img, gt = load_pair(rows["test"][i])
        p = pt[i].astype(np.float32)
        heat = cv2.applyColorMap((p * 255).astype(np.uint8), cv2.COLORMAP_JET)[..., ::-1]
        ov = (0.6 * img + 0.4 * heat).astype(np.uint8)
        c, _ = cv2.findContours((gt > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(ov, c, -1, (0, 255, 0), 1)
        cv2.imwrite(str(vis / f"{rows['test'][i]['label']}_{Path(rows['test'][i]['path']).stem}.png"),
                    np.concatenate([img, heat, ov], 1)[..., ::-1])
    print(json.dumps({k: res[k] for k in ("best_epoch", "test", "test_pixel", "latency_ms_median")}, indent=1))


if __name__ == "__main__":
    main()
