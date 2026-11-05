"""Classification training/evaluation pipeline.

Output layout (``artifacts/checkpoints/<run_id>/``)::

    model.pt            state_dict + metadata (arch, classes, preprocess, temperature)
    ood.npz             Mahalanobis OOD statistics + thresholds
    config.json         full training configuration
    preprocess.json     preprocessing configuration (shared with inference)
    history.json        per-epoch metrics (real values only)
    metrics.json        held-out test metrics + benchmark + calibration
    status.json         live status for the API (running/completed/failed/stopped)
    confusion_matrix.png, history.png, roc_pr.png
    last.pt             resumable state (model + optimizer + scheduler + epoch)

Create ``STOP`` inside the run directory to request a graceful stop after the current batch.
"""
from __future__ import annotations

import copy
import json
import os
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import psutil
import torch
from torch.utils.data import DataLoader

from ml.architectures.registry import (ARCHITECTURES, EmbeddingHook, build_model, count_params, get_module,
                                      set_backbone_trainable)
from ml.config import CHECKPOINT_DIR, EXPERIMENT_DIR, device_info, get_device, seed_everything
from ml.datasets.splitting import load_manifest
from ml.datasets.torch_data import ManifestDataset
from ml.evaluation import metrics as M
from ml.preprocessing.transforms import PreprocessConfig
from ml.training.losses import build_loss
from ml.uncertainty.calibration import MahalanobisOOD, energy_score, fit_temperature, softmax_np


class StopRequested(Exception):
    pass


@dataclass
class TrainConfig:
    arch: str
    dataset: str = "neu_cls"
    run_id: str | None = None
    pretrained: bool = True
    epochs: int = 10
    freeze_epochs: int = 1           # epochs of head-only training (feature extraction) before fine-tuning
    batch_size: int = 32
    lr: float = 1e-3
    backbone_lr_mult: float = 0.1    # discriminative LR for pretrained layers during fine-tuning
    weight_decay: float = 1e-4
    optimizer: str = "adamw"         # adam | adamw
    loss: str = "ce"                 # ce | focal
    label_smoothing: float = 0.05
    class_weighting: bool = True
    dropout: float = 0.2
    patience: int = 3
    seed: int = 42
    num_workers: int = 0
    resume: bool = False
    max_minutes: float = 60.0
    preprocess: dict = field(default_factory=dict)
    tags: dict = field(default_factory=dict)


def _status(run_dir: Path, **kw):
    p = run_dir / "status.json"
    cur = json.loads(p.read_text()) if p.exists() else {}
    cur.update(kw, updated_at=datetime.now(timezone.utc).isoformat())
    p.write_text(json.dumps(cur, indent=2))


@torch.no_grad()
def predict(model, loader, device, arch: str):
    model.eval()
    hook = EmbeddingHook(model, arch)
    logits, ys, embs = [], [], []
    for x, y in loader:
        out = model(x.to(device))
        logits.append(out.cpu().numpy()); ys.append(y.numpy()); embs.append(hook.value.cpu().numpy())
    hook.remove()
    return np.concatenate(logits), np.concatenate(ys), np.concatenate(embs)


def benchmark(model: torch.nn.Module, input_size: int, device, n: int = 20, batch: int = 16) -> dict:
    """Measured (not estimated) latency/throughput on the current device."""
    model.eval()
    proc = psutil.Process(os.getpid())
    x1 = torch.randn(1, 3, input_size, input_size, device=device)
    xb = torch.randn(batch, 3, input_size, input_size, device=device)
    rss0 = proc.memory_info().rss
    with torch.no_grad():
        for _ in range(3):
            model(x1)
        ts = []
        for _ in range(n):
            t = time.perf_counter(); model(x1); ts.append(time.perf_counter() - t)
        t = time.perf_counter()
        for _ in range(3):
            model(xb)
        tb = (time.perf_counter() - t) / 3
    rss1 = proc.memory_info().rss
    params = count_params(model)
    return {
        "device": str(device), "threads": torch.get_num_threads(),
        "latency_ms_median": float(np.median(ts) * 1000), "latency_ms_p95": float(np.percentile(ts, 95) * 1000),
        "throughput_img_s_batch16": float(batch / tb), "params": int(params),
        "param_size_mb": float(params * 4 / 1e6), "rss_delta_mb_inference": float((rss1 - rss0) / 1e6),
    }


def train(cfg: TrainConfig) -> dict:
    seed_everything(cfg.seed)
    device = get_device()
    spec = ARCHITECTURES[cfg.arch]
    run_id = cfg.run_id or f"{cfg.dataset}-{cfg.arch}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    run_dir = CHECKPOINT_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "STOP").unlink(missing_ok=True)
    _status(run_dir, state="running", run_id=run_id, arch=cfg.arch, epoch=0, epochs=cfg.epochs, message="starting")

    try:
        manifest = load_manifest(cfg.dataset)
        classes = sorted({r["label"] for r in manifest["samples"]})
        rows = {s: [r for r in manifest["samples"] if r["split"] == s] for s in ("train", "val", "test")}
        pp = PreprocessConfig.from_dict({"input_size": spec.default_input, **manifest["meta"].get("preprocess", {}),
                                         **cfg.preprocess})
        (run_dir / "config.json").write_text(json.dumps({**cfg.__dict__, "run_id": run_id,
                                                         "dataset_version": manifest["version"],
                                                         "classes": classes, "device": device_info()}, indent=2))
        (run_dir / "preprocess.json").write_text(json.dumps(pp.to_dict(), indent=2))

        ds_tr = ManifestDataset(rows["train"], classes, pp, train=True)
        ds_va = ManifestDataset(rows["val"], classes, pp, train=False)
        ds_te = ManifestDataset(rows["test"], classes, pp, train=False)
        ds_tr_eval = ManifestDataset(rows["train"], classes, pp, train=False)
        g = torch.Generator().manual_seed(cfg.seed)
        dl_tr = DataLoader(ds_tr, cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, generator=g, drop_last=False)
        dl = lambda ds: DataLoader(ds, 64, shuffle=False, num_workers=cfg.num_workers)  # noqa: E731

        model = build_model(cfg.arch, len(classes), pretrained=cfg.pretrained and cfg.arch != "custom_cnn",
                            dropout=cfg.dropout).to(device)

        counts = np.bincount(ds_tr.targets(), minlength=len(classes))
        cw = torch.tensor(len(ds_tr) / (len(classes) * np.maximum(counts, 1)), dtype=torch.float32, device=device) \
            if cfg.class_weighting else None
        crit = build_loss(cfg.loss, cw, cfg.label_smoothing)

        head_ids = {id(p) for p in get_module(model, spec.head_attr).parameters()}
        is_pretrained = cfg.pretrained and cfg.arch != "custom_cnn"
        bb_lr = cfg.lr * (cfg.backbone_lr_mult if is_pretrained else 1.0)
        groups = [{"params": [p for p in model.parameters() if id(p) not in head_ids], "lr": bb_lr},
                  {"params": [p for p in model.parameters() if id(p) in head_ids], "lr": cfg.lr}]
        Opt = torch.optim.AdamW if cfg.optimizer == "adamw" else torch.optim.Adam
        opt = Opt(groups, weight_decay=cfg.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, cfg.epochs))

        start_epoch, history, best_loss, best_state, bad = 1, [], float("inf"), None, 0
        if cfg.resume and (run_dir / "last.pt").exists():
            st = torch.load(run_dir / "last.pt", map_location=device, weights_only=False)
            model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
            start_epoch, history, best_loss, bad = st["epoch"] + 1, st["history"], st["best_loss"], st["bad"]
            best_state = st.get("best_state")
            _status(run_dir, message=f"resumed from epoch {st['epoch']}")

        t0 = time.time()
        for epoch in range(start_epoch, cfg.epochs + 1):
            frozen = is_pretrained and epoch <= cfg.freeze_epochs
            set_backbone_trainable(model, cfg.arch, not frozen)
            model.train()
            tl, tc, tn = 0.0, 0, 0
            for bi, (x, y) in enumerate(dl_tr):
                if (run_dir / "STOP").exists():
                    raise StopRequested()
                x, y = x.to(device), y.to(device)
                opt.zero_grad(set_to_none=True)
                out = model(x)
                loss = crit(out, y)
                loss.backward()
                opt.step()
                tl += loss.item() * len(y); tc += (out.argmax(1) == y).sum().item(); tn += len(y)
                if bi % 10 == 0:
                    _status(run_dir, epoch=epoch, batch=bi, batches=len(dl_tr), message="training")
            sched.step()
            lg, yv, _ = predict(model, dl(ds_va), device, cfg.arch)
            vloss = float(torch.nn.functional.cross_entropy(torch.tensor(lg), torch.tensor(yv)).item())
            vacc = float((lg.argmax(1) == yv).mean())
            h = {"epoch": epoch, "train_loss": tl / tn, "train_acc": tc / tn, "val_loss": vloss, "val_acc": vacc,
                 "lr": opt.param_groups[-1]["lr"], "backbone_frozen": frozen, "elapsed_s": time.time() - t0}
            history.append(h)
            print(f"[{run_id}] epoch {epoch}/{cfg.epochs} train_loss={h['train_loss']:.4f} "
                  f"train_acc={h['train_acc']:.4f} val_loss={vloss:.4f} val_acc={vacc:.4f} "
                  f"t={h['elapsed_s']:.0f}s", flush=True)
            if vloss < best_loss - 1e-4:
                best_loss, best_state, bad = vloss, copy.deepcopy(model.state_dict()), 0
            else:
                bad += 1
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "epoch": epoch, "history": history, "best_loss": best_loss, "bad": bad,
                        "best_state": best_state}, run_dir / "last.pt")
            (run_dir / "history.json").write_text(json.dumps(history, indent=2))
            _status(run_dir, epoch=epoch, val_acc=vacc, val_loss=vloss, message="epoch complete")
            if bad >= cfg.patience:
                print(f"[{run_id}] early stopping at epoch {epoch}", flush=True)
                break
            if (time.time() - t0) / 60 > cfg.max_minutes:
                print(f"[{run_id}] time budget reached at epoch {epoch}", flush=True)
                break
        train_time = time.time() - t0

        model.load_state_dict(best_state)
        # ---- calibration + OOD on train/val only (test untouched)
        lg_va, y_va, e_va = predict(model, dl(ds_va), device, cfg.arch)
        T = fit_temperature(lg_va, y_va)
        _, y_tr, e_tr = predict(model, dl(ds_tr_eval), device, cfg.arch)
        ood = MahalanobisOOD().fit(e_tr, y_tr, len(classes))
        ood.calibrate(ood.score(e_va), energy_score(lg_va, T), quantile=0.99)

        # ---- held-out test
        lg_te, y_te, e_te = predict(model, dl(ds_te), device, cfg.arch)
        probs_te = softmax_np(lg_te, T)
        rep = M.classification_report(y_te, probs_te, classes)
        rep_uncal = M.classification_report(y_te, softmax_np(lg_te, 1.0), classes)
        rep["ece_uncalibrated"] = rep_uncal["ece"]
        rep["temperature"] = T
        te_ood = ood.score(e_te)
        rep["ood_in_distribution_test_flag_rate"] = float((te_ood > ood.threshold).mean())
        curv = M.curves(y_te, probs_te, classes)
        bench = benchmark(model, pp.input_size, device)
        torch.save({"arch": cfg.arch, "classes": classes, "state_dict": model.state_dict(),
                    "preprocess": pp.to_dict(), "temperature": T, "run_id": run_id,
                    "dataset": cfg.dataset, "dataset_version": manifest["version"]}, run_dir / "model.pt")
        np.savez(run_dir / "ood.npz", means=ood.means, prec=ood.prec, threshold=ood.threshold,
                 energy_threshold=ood.energy_threshold)
        np.savez_compressed(run_dir / "test_predictions.npz", logits=lg_te, y=y_te,
                            paths=np.array([r["path"] for r in rows["test"]]))
        size_mb = (run_dir / "model.pt").stat().st_size / 1e6
        result = {"run_id": run_id, "arch": cfg.arch, "dataset": cfg.dataset, "dataset_version": manifest["version"],
                  "classes": classes, "split_sizes": {k: len(v) for k, v in rows.items()},
                  "epochs_run": len(history), "best_val_loss": best_loss, "train_time_s": train_time,
                  "test": rep, "curves": curv, "benchmark": bench, "checkpoint_size_mb": size_mb,
                  "ood": {"threshold": ood.threshold, "energy_threshold": ood.energy_threshold, "quantile": 0.99},
                  "completed_at": datetime.now(timezone.utc).isoformat()}
        M.save_json(result, run_dir / "metrics.json")
        M.plot_confusion(rep["confusion_matrix"], classes, run_dir / "confusion_matrix.png",
                         f"{cfg.arch} - test confusion matrix")
        M.plot_history(history, run_dir / "history.png")
        M.plot_roc_pr(curv, run_dir / "roc_pr.png")
        (run_dir / "last.pt").unlink(missing_ok=True)  # resumable state no longer needed
        _log_mlflow(cfg, run_id, history, result, run_dir)
        _status(run_dir, state="completed", message="completed", test_accuracy=rep["accuracy"],
                test_f1_macro=rep["f1_macro"])
        print(f"[{run_id}] TEST acc={rep['accuracy']:.4f} f1_macro={rep['f1_macro']:.4f} "
              f"latency={bench['latency_ms_median']:.1f}ms", flush=True)
        return result
    except StopRequested:
        _status(run_dir, state="stopped", message="stopped by user (resumable via resume=True)")
        print(f"[{run_id}] stopped", flush=True)
        return {"run_id": run_id, "state": "stopped"}
    except Exception as exc:
        _status(run_dir, state="failed", message=str(exc), traceback=traceback.format_exc())
        raise


def _log_mlflow(cfg: TrainConfig, run_id: str, history: list, result: dict, run_dir: Path):
    try:
        os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
        import mlflow

        mlflow.set_tracking_uri("sqlite:///" + (EXPERIMENT_DIR / "mlflow.db").resolve().as_posix())
        exp_name = f"qualityguard-{cfg.dataset}"
        if mlflow.get_experiment_by_name(exp_name) is None:
            mlflow.create_experiment(exp_name, artifact_location=(EXPERIMENT_DIR / "mlartifacts").resolve().as_uri())
        mlflow.set_experiment(exp_name)
        with mlflow.start_run(run_name=run_id):
            mlflow.log_params({k: v for k, v in cfg.__dict__.items() if isinstance(v, (int, float, str, bool))})
            mlflow.log_param("dataset_version", result["dataset_version"])
            for h in history:
                mlflow.log_metrics({k: v for k, v in h.items() if k not in ("epoch", "backbone_frozen")}, step=h["epoch"])
            t = result["test"]
            mlflow.log_metrics({"test_accuracy": t["accuracy"], "test_f1_macro": t["f1_macro"],
                                "test_ece": t["ece"], "latency_ms": result["benchmark"]["latency_ms_median"]})
            for f in ("metrics.json", "confusion_matrix.png", "history.png", "roc_pr.png", "preprocess.json"):
                mlflow.log_artifact(str(run_dir / f))
    except Exception as exc:  # MLflow is optional
        print(f"MLflow logging skipped: {exc}", flush=True)
