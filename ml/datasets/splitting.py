"""Leakage-aware dataset splitting and versioned manifests.

* Exact and near-duplicate images (dHash Hamming distance <= threshold) are merged into the
  same *group* so they can never straddle train/val/test.
* Explicit ``group`` identifiers (e.g. a physical part id) are honoured.
* Splits are stratified by label at the group level.
* A manifest's ``version`` is the SHA-256 of its sorted (path, label, split) rows, so any change
  to the data or the split yields a new dataset version.
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ml.config import PROCESSED_DIR


class _UF:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        self.p[self.find(a)] = self.find(b)


def build_groups(samples: list[dict], duplicate_pairs: list[list[str]] | None = None) -> list[str]:
    """Union samples that share an explicit group id or appear in a confirmed duplicate pair/group."""
    n = len(samples)
    uf = _UF(n)
    by_group = defaultdict(list)
    for i, s in enumerate(samples):
        if s.get("group"):
            by_group[s["group"]].append(i)
    for idx in by_group.values():
        for j in idx[1:]:
            uf.union(idx[0], j)
    pos = {s["path"]: i for i, s in enumerate(samples)}
    for grp in duplicate_pairs or []:
        idx = [pos[p] for p in grp if p in pos]
        for j in idx[1:]:
            uf.union(idx[0], j)
    return [f"g{uf.find(i)}" for i in range(n)]


def stratified_group_split(samples: list[dict], groups: list[str], ratios=(0.7, 0.15, 0.15), seed: int = 42) -> list[str]:
    """Assign 'train'/'val'/'test' per group, stratified by the group's majority label."""
    rng = np.random.default_rng(seed)
    g2idx = defaultdict(list)
    for i, g in enumerate(groups):
        g2idx[g].append(i)
    by_label = defaultdict(list)
    for g, idx in g2idx.items():
        labels = [samples[i]["label"] for i in idx]
        by_label[max(set(labels), key=labels.count)].append(g)
    split = [""] * len(samples)
    for label in sorted(by_label):
        gs = sorted(by_label[label])
        rng.shuffle(gs)
        n = len(gs)
        n_tr = int(round(ratios[0] * n))
        n_va = int(round(ratios[1] * n))
        for k, g in enumerate(gs):
            name = "train" if k < n_tr else "val" if k < n_tr + n_va else "test"
            for i in g2idx[g]:
                split[i] = name
    return split


def manifest_version(rows: list[dict]) -> str:
    key = sorted((r["path"], str(r["label"]), r["split"]) for r in rows)
    return hashlib.sha256(json.dumps(key).encode()).hexdigest()[:12]


def write_manifest(name: str, rows: list[dict], meta: dict) -> Path:
    version = manifest_version(rows)
    out_dir = PROCESSED_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = defaultdict(lambda: defaultdict(int))
    for r in rows:
        counts[r["split"]][r["label"]] += 1
    doc = {
        "name": name,
        "version": version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "num_samples": len(rows),
        "split_counts": {k: dict(sorted(v.items())) for k, v in sorted(counts.items())},
        "meta": meta,
        "samples": rows,
    }
    path = out_dir / f"manifest_{version}.json"
    path.write_text(json.dumps(doc, indent=1))
    (out_dir / "manifest_latest.json").write_text(json.dumps(doc, indent=1))
    return path


def load_manifest(name_or_path: str | Path) -> dict:
    p = Path(name_or_path)
    if not p.suffix:
        p = PROCESSED_DIR / str(name_or_path) / "manifest_latest.json"
    return json.loads(p.read_text())
