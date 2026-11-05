"""Train + evaluate the component recogniser (automatic product-category routing and unsupported-image rejection).

    python scripts/train_recognizer.py

Data protocol (per domain, from the existing versioned manifests - no test leakage):
  * fit       = manifest 'train' split + first half of 'val'    (normal AND defective images where available,
                because a defective metal nut is still a metal nut)
  * calibrate = second half of 'val'                             (per-domain distance thresholds)
  * test      = manifest 'test' split                            (reported, never used for fitting or thresholds)
  * foreign   = images of categories the system does NOT support (Imagenette natural photos, unused MVTec AD
                categories). Used ONLY for reporting the rejection rate, never for fitting or thresholds.

Output: artifacts/checkpoints/component_recognizer-resnet18-v1/{recognizer.pt, metrics.json, status.json}
"""
from __future__ import annotations

import json
import sys
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from ml.config import CHECKPOINT_DIR, DATA_DIR, absp, seed_everything  # noqa: E402
from ml.datasets.analysis import read_image  # noqa: E402
from ml.datasets.splitting import load_manifest  # noqa: E402
from ml.recognition.router import ComponentRecognizer, Embedder  # noqa: E402

RUN_ID = "component_recognizer-resnet18-v1"
# domain -> (manifest, profile id, display title)
DOMAINS = {
    "steel_strip": ("neu_cls", "steel_strip", "Hot-rolled steel strip surface"),
    "bottle": ("mvtec_bottle", "bottle", "Bottle"),
    "metal_nut": ("mvtec_metal_nut", "metal_nut", "Metal nut"),
    "screw": ("mvtec_screw", "screw", "Screw"),
    "metal_grid": ("mvtec_grid", "metal_grid", "Metal grid / mesh"),
    "commutator": ("ksdd2", "commutator_surface", "Electrical commutator surface"),
    **{f"mpdd_{c}": (f"mpdd_{c}", f"mpdd_{c}", t) for c, t in (
        ("bracket_black", "Metal bracket (black)"), ("bracket_brown", "Metal bracket (brown)"),
        ("bracket_white", "Metal bracket (white)"), ("connector", "Metal connector"),
        ("metal_plate", "Metal plate"), ("tubes", "Metal tubes"))},
}
MAX_FIT_PER_DOMAIN = 400


def rows_for(manifest: str, rng) -> dict[str, list[dict]]:
    m = load_manifest(manifest)
    s = {k: [r for r in m["samples"] if r["split"] == k] for k in ("train", "val", "test")}
    val = [s["val"][i] for i in rng.permutation(len(s["val"]))]
    fit = s["train"] + val[: len(val) // 2]
    if len(fit) > MAX_FIT_PER_DOMAIN:
        fit = [fit[i] for i in rng.choice(len(fit), MAX_FIT_PER_DOMAIN, replace=False)]
    return {"fit": fit, "calib": val[len(val) // 2:], "test": s["test"], "version": m["version"]}


def embed_paths(emb: Embedder, paths: list[Path], bs: int = 32) -> np.ndarray:
    out = []
    for i in range(0, len(paths), bs):
        imgs = [read_image(p) for p in paths[i:i + bs]]
        out.append(emb.embed_bgr(imgs))
        print(f"    embedded {min(i + bs, len(paths))}/{len(paths)}", end="\r", flush=True)
    print()
    return np.concatenate(out) if out else np.zeros((0, 768), np.float32)


def foreign_sets(rng) -> dict[str, list[Path]]:
    sets: dict[str, list[Path]] = {}
    tgz = DATA_DIR / "downloads" / "imagenette" / "imagenette2-160.tgz"
    dest = DATA_DIR / "raw" / "foreign_probe" / "imagenette"
    if tgz.exists() and not dest.exists():
        dest.mkdir(parents=True)
        with tarfile.open(tgz) as t:
            members = [m for m in t.getmembers() if m.isfile() and "/val/" in m.name and m.name.endswith(".JPEG")]
            pick = [members[i] for i in rng.choice(len(members), 300, replace=False)]
            for m in pick:  # extract file bytes only (no paths from the archive are used -> no traversal)
                (dest / Path(m.name).name).write_bytes(t.extractfile(m).read())
    if dest.exists():
        sets["imagenette_natural_photos"] = sorted(dest.glob("*.JPEG"))
    mv = DATA_DIR / "raw" / "foreign_probe" / "mvtec_unsupported"
    if mv.exists():
        sets["mvtec_unsupported_categories"] = sorted(mv.glob("*.png"))
    return sets


def main():
    seed_everything(42)
    rng = np.random.default_rng(42)
    t0 = time.time()
    emb = Embedder()
    data, versions = {}, {}
    for d, (man, _, _) in DOMAINS.items():
        try:
            data[d] = rows_for(man, rng)
            versions[d] = data[d].pop("version")
        except FileNotFoundError:
            print(f"!! manifest {man} missing - domain '{d}' skipped (run prepare_datasets.py)")
    E = {split: {} for split in ("fit", "calib", "test")}
    for d, parts in data.items():
        for split, rows in parts.items():
            print(f"  {d}/{split}: {len(rows)}")
            E[split][d] = embed_paths(emb, [absp(r["path"]) for r in rows])

    def stack(split):
        xs, ys = [], []
        for d, e in E[split].items():
            xs.append(e); ys += [d] * len(e)
        return np.concatenate(xs), ys

    rec = ComponentRecognizer()
    fit_info = rec.fit(*stack("fit"))
    calib_info = rec.calibrate(*stack("calib"))
    rec.domain_profiles = {d: DOMAINS[d][1] for d in rec.domains}
    rec.domain_titles = {d: DOMAINS[d][2] for d in rec.domains}

    # ---- held-out in-domain test
    Xt, yt = stack("test")
    dec = rec.decide_batch(Xt)
    pred = [x["domain"] for x in dec]
    sup = np.array([x["supported"] for x in dec])
    yt_a, pred_a = np.array(yt), np.array(pred)
    per = {}
    for d in rec.domains:
        m = yt_a == d
        per[d] = {"n": int(m.sum()), "top1_accuracy": float((pred_a[m] == d).mean()),
                  "accepted_and_correct": float(((pred_a[m] == d) & sup[m]).mean()),
                  "false_reject_rate": float((~sup[m]).mean()),
                  "accepted_but_wrong_domain": int(((pred_a[m] != d) & sup[m]).sum())}
    cm = {t: {p: int(((yt_a == t) & (pred_a == p)).sum()) for p in rec.domains} for t in rec.domains}
    test = {"n": int(len(yt)), "top1_accuracy": float((pred_a == yt_a).mean()),
            "false_reject_rate": float((~sup).mean()),
            "accepted_but_wrong_domain": int(((pred_a != yt_a) & sup).sum()),
            "per_domain": per, "confusion_matrix": cm}

    # ---- foreign (unsupported) probes - evaluation only
    foreign = {}
    for name, paths in foreign_sets(rng).items():
        fe = embed_paths(emb, paths)
        fd = rec.decide_batch(fe)
        acc = [x for x in fd if x["supported"]]
        foreign[name] = {"n": len(paths), "rejected_rate": float(1 - len(acc) / max(1, len(paths))),
                         "accepted_as": {d: sum(1 for x in acc if x["domain"] == d) for d in rec.domains if
                                         any(x["domain"] == d for x in acc)}}

    res = {"run_id": RUN_ID, "task": "component_recognition", "completed_at": datetime.now(timezone.utc).isoformat(),
           "embedder": rec.state()["embedder"], "classifier": "multinomial logistic regression (C=2, balanced)",
           "rejection_rule": f"supported iff probability >= {rec.min_probability} AND {rec.k}-NN cosine distance <= "
                             "per-domain threshold (1.10 x 99th percentile of held-out calibration distances)",
           "domains": {d: {"manifest": DOMAINS[d][0], "profile_id": DOMAINS[d][1], "title": DOMAINS[d][2]}
                       for d in rec.domains},
           "dataset_versions": versions, "fit": fit_info, "calibration": calib_info, "test": test,
           "foreign_probes": foreign, "total_time_s": time.time() - t0,
           "limitations": "Recognises only the trained domains. Rejection of unsupported images is measured on the "
                          "listed probe sets only; other kinds of images may behave differently."}
    out = CHECKPOINT_DIR / RUN_ID
    out.mkdir(parents=True, exist_ok=True)
    import torch
    torch.save(rec.state(), out / "recognizer.pt")
    (out / "metrics.json").write_text(json.dumps(res, indent=2))
    (out / "status.json").write_text(json.dumps({"state": "completed", "run_id": RUN_ID}))
    print(json.dumps({"test": {k: v for k, v in test.items() if k != "confusion_matrix"}, "foreign": foreign}, indent=1))


if __name__ == "__main__":
    main()
