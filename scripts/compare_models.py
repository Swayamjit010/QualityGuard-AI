"""Build the model comparison (proposal objectives 5-7) from artifacts/checkpoints/*/metrics.json.

    python scripts/compare_models.py

Writes:
    artifacts/metrics/model_comparison.json
    artifacts/metrics/model_comparison.md        (Markdown tables, pasted into docs/RESULTS.md)
    artifacts/visualizations/model_comparison.png
    docs/RESULTS.md                              (tables between the GENERATED markers are refreshed)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ml.config import CHECKPOINT_DIR, METRICS_DIR, VIS_DIR  # noqa: E402


def load():
    clf, anom = [], []
    for d in sorted(CHECKPOINT_DIR.iterdir()):
        f = d / "metrics.json"
        if not f.exists():
            continue
        m = json.loads(f.read_text())
        if (d / "anomaly.pt").exists() and "engines" in m:
            anom.append(m)
        elif (d / "model.pt").exists() and "test" in m and m.get("dataset", "neu_cls") == "neu_cls":
            # the comparison table is only meaningful on one shared held-out split (NEU-CLS);
            # KolektorSDD2 classifiers, segmentation and recognition are reported in RESULTS.md section 10
            clf.append(m)
    return clf, anom


def fmt(v, nd=4):
    return "—" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def main():
    clf, anom = load()
    rows = []
    for m in clf:
        t, b = m["test"], m.get("benchmark", {})
        pc = t.get("per_class", {})
        rows.append({"run_id": m["run_id"], "arch": m.get("arch") or m["run_id"].split("-")[1],
                     "pretrained": m.get("config", {}).get("pretrained", m.get("arch") != "custom_cnn"),
                     "epochs": m.get("epochs_run"), "accuracy": t.get("accuracy"),
                     "precision_macro": t.get("precision_macro"), "recall_macro": t.get("recall_macro"),
                     "f1_macro": t.get("f1_macro"), "f1_weighted": t.get("f1_weighted"),
                     "roc_auc": t.get("roc_auc_macro_ovr"), "ece": t.get("ece"),
                     "min_class_recall": min((v["recall"] for v in pc.values()), default=None),
                     "latency_ms": b.get("latency_ms_median"), "params_mb": b.get("param_size_mb"),
                     "train_time_min": (m.get("train_time_s") or 0) / 60})
    rows.sort(key=lambda r: (-(r["f1_macro"] or 0), r["latency_ms"] or 1e9))
    an_rows = []
    for m in anom:
        if str(m.get("category", "")).startswith("custom_"):
            continue  # user-added products: reported in RESULTS.md section 9
        for eng, e in m["engines"].items():
            an_rows.append({"run_id": m["run_id"], "category": m.get("category"), "engine": eng,
                            "image_auroc": e["test"].get("image_auroc"), "recall": e["test"].get("recall"),
                            "fpr": e["test"].get("false_positive_rate"),
                            "pixel_auroc": e.get("test_pixel", {}).get("pixel_auroc"),
                            "dice": e.get("test_pixel", {}).get("mean_dice_defective"),
                            "latency_ms": m.get("latency_ms_median", {}).get(eng)})
        f = m.get("fusion", {})
        label = "anomaly-engine fusion" + (" (superseded: all engines voting)" if f.get("superseded") else "")
        an_rows.append({"run_id": m["run_id"], "category": m.get("category"), "engine": label,
                        "auto_accuracy": f.get("auto_accuracy"), "review_rate": f.get("review_rate"),
                        "missed": f.get("missed_defects_auto_pass"), "false_rejects": f.get("false_rejects_auto_fail")})

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    (METRICS_DIR / "model_comparison.json").write_text(json.dumps({"classifiers": rows, "anomaly": an_rows}, indent=2))

    md = ["### Classifiers - NEU-CLS held-out test split (identical for all models)", "",
          "| Model | Pre-trained | Epochs | Accuracy | Macro P | Macro R | Macro F1 | ROC-AUC | Worst-class recall | ECE | CPU latency (ms) | Size (MB) | Train (min) |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['arch']} | {'yes' if r['pretrained'] else 'no (baseline)'} | {r['epochs']} | {fmt(r['accuracy'])} | "
                  f"{fmt(r['precision_macro'])} | {fmt(r['recall_macro'])} | {fmt(r['f1_macro'])} | {fmt(r['roc_auc'])} | "
                  f"{fmt(r['min_class_recall'], 3)} | {fmt(r['ece'], 3)} | {fmt(r['latency_ms'], 1)} | "
                  f"{fmt(r['params_mb'], 1)} | {fmt(r['train_time_min'], 1)} |")
    md += ["", "### Anomaly detectors - MVTec AD and KolektorSDD2 (commutator) test splits", "",
           "The fusion rows describe the anomaly engines on their own. The decision actually deployed for each category "
           "(including validation-based engine selection and, for the commutator, classifier + segmentation voting) is "
           "measured end to end in `reports/end_to_end_evaluation.md`.", "",
           "| Category | Engine | Image AUROC | Recall | FPR | Pixel AUROC | Dice | Latency (ms) |",
           "|---|---|---|---|---|---|---|---|"]
    for r in an_rows:
        if "auto_accuracy" in r:
            md.append(f"| {r['category']} | {r['engine']} | auto-decided accuracy {fmt(r['auto_accuracy'], 3)} | "
                      f"missed defects {r['missed']} | false rejects {r['false_rejects']} | review rate {fmt(r['review_rate'], 3)} | | |")
        else:
            md.append(f"| {r['category']} | {r['engine']} | {fmt(r['image_auroc'])} | {fmt(r['recall'], 3)} | "
                      f"{fmt(r['fpr'], 3)} | {fmt(r['pixel_auroc'])} | {fmt(r['dice'], 3)} | {fmt(r['latency_ms'], 0)} |")
    (METRICS_DIR / "model_comparison.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    res = ROOT / "docs" / "RESULTS.md"  # refresh the generated block between the markers
    if res.exists():
        doc = res.read_text(encoding="utf-8")
        a, b = "<!-- BEGIN GENERATED TABLES -->", "<!-- END GENERATED TABLES -->"
        if a in doc and b in doc:
            doc = doc[:doc.index(a) + len(a)] + "\n" + "\n".join(md) + "\n" + doc[doc.index(b):]
            res.write_text(doc, encoding="utf-8")

    if rows:
        VIS_DIR.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
        names = [r["arch"] for r in rows]
        ax[0].bar(names, [r["f1_macro"] or 0 for r in rows], color="#1d6fe0")
        ax[0].set_ylim(min(0.8, min(r["f1_macro"] or 0 for r in rows) - 0.02), 1.0)
        ax[0].set_title("Macro F1 (test)")
        for i, r in enumerate(rows):
            ax[0].text(i, (r["f1_macro"] or 0), f"{r['f1_macro']:.3f}", ha="center", va="bottom", fontsize=9)
        ax[1].scatter([r["latency_ms"] for r in rows], [r["f1_macro"] for r in rows],
                      s=[max(30, (r["params_mb"] or 1) * 6) for r in rows], color="#14b8a6", alpha=0.8)
        for r in rows:
            ax[1].annotate(r["arch"], (r["latency_ms"], r["f1_macro"]), textcoords="offset points", xytext=(6, 4), fontsize=9)
        ax[1].set_xlabel("CPU latency per image (ms, median)")
        ax[1].set_ylabel("Macro F1")
        ax[1].set_title("Accuracy vs. cost (bubble = model size)")
        ax[1].grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(VIS_DIR / "model_comparison.png", dpi=130)
    print("\n".join(md))


if __name__ == "__main__":
    main()
