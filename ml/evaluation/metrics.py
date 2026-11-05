"""Classification / anomaly / segmentation metrics and plots. All computed from real predictions."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn import metrics as skm  # noqa: E402


def classification_report(y_true: np.ndarray, probs: np.ndarray, classes: list[str]) -> dict:
    y_pred = probs.argmax(1)
    k = len(classes)
    out = {
        "num_samples": int(len(y_true)),
        "accuracy": float(skm.accuracy_score(y_true, y_pred)),
        "precision_macro": float(skm.precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(skm.recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(skm.f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_weighted": float(skm.precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(skm.recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(skm.f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "confusion_matrix": skm.confusion_matrix(y_true, y_pred, labels=list(range(k))).tolist(),
        "classes": classes,
    }
    p, r, f, s = skm.precision_recall_fscore_support(y_true, y_pred, labels=list(range(k)), zero_division=0)
    out["per_class"] = {c: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
                        for i, c in enumerate(classes)}
    try:
        y1h = np.eye(k)[y_true]
        out["roc_auc_macro_ovr"] = float(skm.roc_auc_score(y1h, probs, average="macro", multi_class="ovr"))
        out["pr_auc_macro"] = float(skm.average_precision_score(y1h, probs, average="macro"))
        out["per_class_roc_auc"] = {c: float(skm.roc_auc_score(y1h[:, i], probs[:, i])) for i, c in enumerate(classes)}
    except ValueError:
        pass
    # expected calibration error (15 bins)
    conf = probs.max(1)
    correct = (y_pred == y_true).astype(float)
    bins = np.linspace(0, 1, 16)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    out["ece"] = float(ece)
    return out


def binary_anomaly_report(y_true: np.ndarray, scores: np.ndarray, threshold: float | None = None) -> dict:
    """y_true: 1 = anomalous/defective, 0 = normal."""
    out = {"num_samples": int(len(y_true)), "num_anomalous": int(y_true.sum()),
           "image_auroc": float(skm.roc_auc_score(y_true, scores)),
           "image_ap": float(skm.average_precision_score(y_true, scores))}
    if threshold is not None:
        pred = (scores >= threshold).astype(int)
        tn, fp, fn, tp = skm.confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        out.update({
            "threshold": float(threshold), "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
            "accuracy": float((tp + tn) / len(y_true)),
            "precision": float(tp / max(1, tp + fp)), "recall": float(tp / max(1, tp + fn)),
            "f1": float(skm.f1_score(y_true, pred, zero_division=0)),
            "false_negative_rate": float(fn / max(1, tp + fn)), "false_positive_rate": float(fp / max(1, fp + tn)),
        })
    return out


def threshold_for_f1(y_true: np.ndarray, scores: np.ndarray) -> float:
    p, r, t = skm.precision_recall_curve(y_true, scores)
    f1 = 2 * p * r / np.clip(p + r, 1e-12, None)
    return float(t[np.nanargmax(f1[:-1])])


def threshold_for_recall(y_true: np.ndarray, scores: np.ndarray, min_recall: float = 0.95) -> float:
    """Highest threshold that still achieves >= min_recall on defects (prioritises low false negatives)."""
    pos = np.sort(scores[y_true == 1])
    k = int(np.floor((1 - min_recall) * len(pos)))
    return float(pos[k]) if len(pos) else float("inf")


def pixel_metrics(masks: np.ndarray, maps: np.ndarray, threshold: float | None = None, max_pixels: int = 4_000_000) -> dict:
    """masks/maps: (N,H,W). Pixel AUROC, AP, and Dice/IoU at threshold."""
    y = masks.reshape(-1).astype(np.uint8)
    s = maps.reshape(-1)
    if len(y) > max_pixels:
        idx = np.random.default_rng(0).choice(len(y), max_pixels, replace=False)
        y, s = y[idx], s[idx]
    out = {"pixel_auroc": float(skm.roc_auc_score(y, s)) if 0 < y.sum() < len(y) else None,
           "pixel_ap": float(skm.average_precision_score(y, s)) if y.sum() else None}
    if threshold is not None:
        pred = maps >= threshold
        gt = masks > 0
        inter = np.logical_and(pred, gt).sum((1, 2))
        union = np.logical_or(pred, gt).sum((1, 2))
        psum = pred.sum((1, 2)) + gt.sum((1, 2))
        valid = gt.sum((1, 2)) > 0
        out["mean_iou_defective"] = float((inter[valid] / np.maximum(union[valid], 1)).mean()) if valid.any() else None
        out["mean_dice_defective"] = float((2 * inter[valid] / np.maximum(psum[valid], 1)).mean()) if valid.any() else None
        out["pixel_threshold"] = float(threshold)
    return out


# ---------------------------------------------------------------- plots
def plot_confusion(cm, classes, path: Path, title="Confusion matrix"):
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(1.1 * len(classes) + 2, 1.0 * len(classes) + 1.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=9)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    fig.colorbar(im, fraction=0.046); fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_history(history: list[dict], path: Path):
    ep = [h["epoch"] for h in history]
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
    axs[0].plot(ep, [h["train_loss"] for h in history], label="train"); axs[0].plot(ep, [h["val_loss"] for h in history], label="val")
    axs[0].set_title("Loss"); axs[0].set_xlabel("epoch"); axs[0].legend()
    axs[1].plot(ep, [h["train_acc"] for h in history], label="train"); axs[1].plot(ep, [h["val_acc"] for h in history], label="val")
    axs[1].set_title("Accuracy"); axs[1].set_xlabel("epoch"); axs[1].legend()
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def curves(y_true: np.ndarray, probs: np.ndarray, classes: list[str]) -> dict:
    """ROC/PR curve points per class (downsampled) for the frontend."""
    k = len(classes)
    y1h = np.eye(k)[y_true]
    out = {"roc": {}, "pr": {}}
    for i, c in enumerate(classes):
        fpr, tpr, _ = skm.roc_curve(y1h[:, i], probs[:, i])
        pr, rc, _ = skm.precision_recall_curve(y1h[:, i], probs[:, i])
        sel = np.linspace(0, len(fpr) - 1, min(60, len(fpr))).astype(int)
        sel2 = np.linspace(0, len(pr) - 1, min(60, len(pr))).astype(int)
        out["roc"][c] = [[float(fpr[j]), float(tpr[j])] for j in sel]
        out["pr"][c] = [[float(rc[j]), float(pr[j])] for j in sel2]
    return out


def plot_roc_pr(curv: dict, path: Path):
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    for c, pts in curv["roc"].items():
        a = np.array(pts); axs[0].plot(a[:, 0], a[:, 1], label=c)
    axs[0].plot([0, 1], [0, 1], "k--", lw=0.8); axs[0].set_title("ROC (one-vs-rest)"); axs[0].set_xlabel("FPR"); axs[0].set_ylabel("TPR")
    for c, pts in curv["pr"].items():
        a = np.array(pts); axs[1].plot(a[:, 0], a[:, 1], label=c)
    axs[1].set_title("Precision-Recall"); axs[1].set_xlabel("Recall"); axs[1].set_ylabel("Precision"); axs[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def save_json(obj, path: Path):
    Path(path).write_text(json.dumps(obj, indent=2, default=float))
