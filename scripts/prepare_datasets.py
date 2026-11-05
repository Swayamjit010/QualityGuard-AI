"""Analyse raw datasets, build leakage-aware splits and write versioned manifests + quality reports.

    python scripts/prepare_datasets.py            # all present datasets
    python scripts/prepare_datasets.py neu_cls
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from ml.config import METRICS_DIR, RAW_DIR  # noqa: E402
from ml.datasets.analysis import analyze_samples  # noqa: E402
from ml.datasets.registry import scan  # noqa: E402
from ml.datasets.splitting import build_groups, stratified_group_split, write_manifest  # noqa: E402

NEU_PREPROCESS = {
    "color_mode": "gray3",
    "augment": {"hflip": True, "vflip": True, "rot90": True, "rotate_deg": 0, "zoom": 0.1, "translate": 0.05,
                "brightness": 0.15, "contrast": 0.15, "noise": True, "cutout": False},
}


def save_report(name: str, rep: dict):
    rep = {k: v for k, v in rep.items() if not k.startswith("_")}
    out = METRICS_DIR / f"dataset_report_{name}.json"
    out.write_text(json.dumps(rep, indent=2))
    print(f"  report -> {out}")


def prepare_neu():
    layout, samples = scan(RAW_DIR / "neu_cls", "folder_classification")
    rows = [s.to_dict() for s in samples]
    print(f"NEU-CLS: {len(rows)} samples ({layout})")
    rep = analyze_samples(rows)
    print(f"  corrupted={rep['num_corrupted']} exact_dup_groups={rep['exact_duplicate_groups']} "
          f"near_dup_pairs={rep['near_duplicate_pairs']} sizes={rep['image_sizes']}")
    bad = {c["path"] for c in rep["corrupted"]}
    rows = [r for r in rows if r["path"] not in bad]
    groups = build_groups(rows, rep["_exact_duplicates_all"] + rep["_near_duplicates_all"])
    split = stratified_group_split(rows, groups, (0.7, 0.15, 0.15), seed=42)
    for r, g, s in zip(rows, groups, split):
        r["group"], r["split"] = g, s
    rep["num_groups"] = len(set(groups))
    p = write_manifest("neu_cls", rows, {"task": "classification", "preprocess": NEU_PREPROCESS,
                                         "split_policy": "stratified by label; exact (SHA-256) and pixel-confirmed near duplicates kept in the same split",
                                         "seed": 42, "ratios": [0.7, 0.15, 0.15]})
    rep["manifest"] = str(p.name)
    save_report("neu_cls", rep)
    print(f"  manifest -> {p}")


def prepare_mvtec(category: str, source: str = "mvtec_ad", prefix: str = "mvtec"):
    """MVTec: train/good stays train. Official test is split 50/50 (stratified by defect type) into
    'val' (threshold calibration) and 'test' (held-out reporting). 10% of train/good goes to 'val'
    as additional normal calibration data."""
    root = RAW_DIR / source / category
    if not root.exists():
        print(f"  !! {root} not found - skipped")
        return
    _, samples = scan(root, "mvtec")
    rows = [s.to_dict() for s in samples]
    for r in rows:
        r["meta"]["category"] = category
    print(f"MVTec {category}: {len(rows)} samples")
    rep = analyze_samples(rows)
    rng = np.random.default_rng(42)
    train = [r for r in rows if r["split_hint"] == "train"]
    idx = rng.permutation(len(train))
    for k, i in enumerate(idx):
        train[i]["split"] = "val" if k < int(0.1 * len(train)) else "train"
    test = [r for r in rows if r["split_hint"] == "test"]
    by = {}
    for r in test:
        by.setdefault(r["label"], []).append(r)
    for lbl, lst in by.items():
        perm = rng.permutation(len(lst))
        for k, i in enumerate(perm):
            lst[i]["split"] = "val" if k < len(lst) // 2 else "test"
    p = write_manifest(f"{prefix}_{category}", rows, {
        "task": "anomaly_detection", "category": category,
        "split_policy": "train/good: 90% train, 10% val. official test: 50% val (calibration), 50% test (held-out), "
                        "stratified by defect type", "seed": 42})
    rep["manifest"] = p.name
    save_report(f"{prefix}_{category}", rep)
    print(f"  manifest -> {p}")


def prepare_ksdd2():
    """KolektorSDD2: label = 'defect' if the *_GT.png mask has any positive pixel, else 'good'.
    Official test split is kept as the held-out test set. Official train is split 85/15 into train/val,
    stratified by label, with exact/near duplicates grouped so they never straddle train and val."""
    import cv2

    root = RAW_DIR / "ksdd2"
    if not root.exists():
        return
    rows = []
    for split in ("train", "test"):
        for p in sorted((root / split).glob("*.png")):
            if p.stem.endswith("_GT"):
                continue
            gt = p.with_name(f"{p.stem}_GT.png")
            if not gt.exists():
                print(f"  !! no mask for {p.name} - skipped"); continue
            m = cv2.imread(str(gt), cv2.IMREAD_GRAYSCALE)
            lbl = "defect" if m is not None and (m > 127).any() else "good"
            from ml.config import rel
            rows.append({"path": rel(p), "label": lbl, "split_hint": split, "mask_path": rel(gt) if lbl == "defect" else None,
                         "group": None, "meta": {"category": "commutator"}})
    print(f"KolektorSDD2: {len(rows)} samples")
    rep = analyze_samples(rows)
    print(f"  corrupted={rep['num_corrupted']} exact_dup_groups={rep['exact_duplicate_groups']} "
          f"near_dup_pairs={rep['near_duplicate_pairs']}")
    bad = {c["path"] for c in rep["corrupted"]}
    rows = [r for r in rows if r["path"] not in bad]
    tr = [r for r in rows if r["split_hint"] == "train"]
    groups = build_groups(tr, rep["_exact_duplicates_all"] + rep["_near_duplicates_all"])
    sp = stratified_group_split(tr, groups, (0.85, 0.15, 0.0), seed=42)
    for r, g, s_ in zip(tr, groups, sp):
        r["group"], r["split"] = g, ("val" if s_ == "test" else s_)
    test_paths = {r["path"] for r in rows if r["split_hint"] == "test"}
    tr_paths = {r["path"] for r in tr}
    leaked = [p for grp in rep["_exact_duplicates_all"] for p in grp if set(grp) & test_paths and set(grp) & tr_paths]
    for r in rows:
        if r["split_hint"] == "test":
            r["split"] = "test"
    rep["cross_split_exact_duplicates_official"] = len(leaked)
    p = write_manifest("ksdd2", rows, {
        "task": "binary_classification+segmentation+anomaly_detection", "category": "commutator",
        "preprocess": {"color_mode": "rgb", "augment": {"hflip": True, "vflip": True, "rot90": False, "rotate_deg": 0,
                       "zoom": 0.05, "translate": 0.03, "brightness": 0.15, "contrast": 0.15, "noise": False,
                       "cutout": False}},
        "split_policy": "official test kept as held-out test; official train split 85/15 train/val stratified by "
                        "label with duplicate grouping", "seed": 42,
        "license": "CC BY-NC-SA 4.0", "source": "https://go.vicos.si/kolektorsdd2"})
    rep["manifest"] = p.name
    save_report("ksdd2", rep)
    print(f"  manifest -> {p}")


if __name__ == "__main__":
    want = sys.argv[1:] or ["neu_cls", "mvtec_bottle", "mvtec_metal_nut", "mvtec_screw", "mvtec_grid", "ksdd2"]
    for w in want:
        if w == "neu_cls":
            prepare_neu()
        elif w.startswith("mvtec_"):
            prepare_mvtec(w[len("mvtec_"):])
        elif w == "ksdd2":
            prepare_ksdd2()
        elif w.startswith("mpdd_"):
            prepare_mvtec(w[len("mpdd_"):], source="mpdd", prefix="mpdd")
