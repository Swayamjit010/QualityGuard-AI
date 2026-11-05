"""Component / surface recognition ("which supported product is this, if any?").

Runs BEFORE defect inspection. It answers two questions:

1. Which supported domain (steel strip, bottle, metal nut, screw, ...) does the image most resemble?
   -> multinomial logistic regression on frozen ImageNet ResNet-18 global embeddings.
2. Is the image close enough to that domain's training images to be inspected by its models at all?
   -> k-nearest-neighbour cosine distance to the predicted domain's reference embeddings, compared
      with a per-domain threshold calibrated on held-out *in-domain* images only.

Rejection is deliberately conservative: an image is ``supported`` only if the classifier probability
is >= ``min_probability`` AND the kNN distance is within the domain's calibrated threshold. Anything
else is reported as an unsupported/unfamiliar component and the inspection is routed to manual review.

This is a recogniser for the categories the system was trained on. It is not a general machinery
classifier: an image of a pump, gear or valve that is not among the trained domains is (correctly)
reported as unsupported.
"""
from __future__ import annotations

import threading

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.models as tvm

from ml.preprocessing.transforms import PreprocessConfig, preprocess_for_inference

ROUTER_PP = PreprocessConfig(input_size=224, color_mode="rgb", augment={})


class Embedder:
    """Frozen ImageNet ResNet-18; embedding = L2-normalised concat of GAP(layer3) and GAP(layer4) (768-d)."""

    def __init__(self):
        net = tvm.resnet18(weights=tvm.ResNet18_Weights.DEFAULT).eval()
        self.net = net
        for p in net.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> np.ndarray:
        n = self.net
        x = n.maxpool(n.relu(n.bn1(n.conv1(x))))
        x = n.layer2(n.layer1(x))
        l3 = n.layer3(x)
        l4 = n.layer4(l3)
        g = torch.cat([F.adaptive_avg_pool2d(l3, 1).flatten(1), F.adaptive_avg_pool2d(l4, 1).flatten(1)], 1)
        return F.normalize(g, dim=1).numpy().astype(np.float32)

    def embed_bgr(self, imgs_bgr: list[np.ndarray]) -> np.ndarray:
        xs = torch.stack([preprocess_for_inference(im, ROUTER_PP)[0] for im in imgs_bgr])
        return self(xs)


class ComponentRecognizer:
    def __init__(self, k: int = 5, min_probability: float = 0.80):
        self.k, self.min_probability = k, min_probability
        self.domains: list[str] = []
        self.W = self.b = None               # logistic-regression weights (numpy) - no sklearn at inference
        self.refs: dict[str, np.ndarray] = {}
        self.thresholds: dict[str, float] = {}
        self.domain_profiles: dict[str, str] = {}
        self.domain_titles: dict[str, str] = {}
        self._emb: Embedder | None = None
        self._lock = threading.Lock()

    # ---------------------------------------------------------------- training
    def fit(self, emb: np.ndarray, labels: list[str], C: float = 2.0, seed: int = 42) -> dict:
        from sklearn.linear_model import LogisticRegression

        self.domains = sorted(set(labels))
        y = np.array([self.domains.index(l) for l in labels])
        clf = LogisticRegression(C=C, max_iter=3000, class_weight="balanced", random_state=seed)
        clf.fit(emb, y)
        self.W, self.b = clf.coef_.astype(np.float32), clf.intercept_.astype(np.float32)
        if len(self.domains) == 2:  # sklearn binary shape -> 2-class softmax form
            self.W = np.concatenate([-self.W / 2, self.W / 2]); self.b = np.concatenate([-self.b / 2, self.b / 2])
        self.refs = {d: emb[y == i] for i, d in enumerate(self.domains)}
        return {"train_images": int(len(y)), "per_domain": {d: int((y == i).sum()) for i, d in enumerate(self.domains)}}

    def calibrate(self, emb: np.ndarray, labels: list[str], factor: float = 1.10, q: float = 0.99) -> dict:
        """Per-domain distance threshold = factor x q-quantile of kNN distances of held-out in-domain images."""
        out = {}
        for d in self.domains:
            e = emb[np.array(labels) == d]
            dist = self.knn_distance(e, d)
            self.thresholds[d] = float(np.quantile(dist, q) * factor)
            out[d] = {"n": int(len(e)), "q": q, "factor": factor, "threshold": self.thresholds[d],
                      "max_calib_distance": float(dist.max())}
        return out

    # ---------------------------------------------------------------- inference
    def probs(self, emb: np.ndarray) -> np.ndarray:
        z = emb @ self.W.T + self.b
        z = z - z.max(1, keepdims=True)
        p = np.exp(z)
        return p / p.sum(1, keepdims=True)

    def knn_distance(self, emb: np.ndarray, domain: str) -> np.ndarray:
        ref = self.refs[domain]
        d = 1.0 - emb @ ref.T                      # cosine distance (embeddings are L2-normalised)
        k = min(self.k, ref.shape[0])
        return np.sort(d, 1)[:, :k].mean(1)

    def decide_batch(self, emb: np.ndarray) -> list[dict]:
        P = self.probs(emb)
        out = []
        for i, p in enumerate(P):
            order = np.argsort(-p)
            top = self.domains[int(order[0])]
            dist = float(self.knn_distance(emb[i:i + 1], top)[0])
            thr = self.thresholds[top]
            reasons = []
            if p[order[0]] < self.min_probability:
                reasons.append(f"Recogniser probability {p[order[0]]:.2f} < {self.min_probability:.2f}: the image does not "
                               f"clearly resemble any one supported component category.")
            if dist > thr:
                reasons.append(f"Appearance distance to '{self.domain_titles.get(top, top)}' references {dist:.3f} > "
                               f"calibrated limit {thr:.3f}: the image is unlike the images this category was trained on.")
            out.append({
                "domain": top, "title": self.domain_titles.get(top, top), "profile_id": self.domain_profiles.get(top),
                "probability": float(p[order[0]]), "distance": dist, "distance_threshold": thr,
                "supported": not reasons, "reasons": reasons or ["Recognised as a supported category."],
                "alternatives": [{"domain": self.domains[int(j)], "probability": float(p[j])} for j in order[1:3]],
            })
        return out

    def recognize(self, img_bgr: np.ndarray) -> dict:
        with self._lock:
            if self._emb is None:
                self._emb = Embedder()
            e = self._emb.embed_bgr([img_bgr])
        return self.decide_batch(e)[0]

    # ---------------------------------------------------------------- persistence
    def state(self) -> dict:
        return {"k": self.k, "min_probability": self.min_probability, "domains": self.domains,
                "W": torch.from_numpy(self.W), "b": torch.from_numpy(self.b),
                "refs": {d: torch.from_numpy(r).half() for d, r in self.refs.items()},
                "thresholds": self.thresholds, "domain_profiles": self.domain_profiles,
                "domain_titles": self.domain_titles, "preprocess": ROUTER_PP.to_dict(),
                "embedder": "torchvision resnet18 IMAGENET1K_V1, GAP(layer3)+GAP(layer4), L2-normalised"}

    @classmethod
    def load(cls, s: dict) -> "ComponentRecognizer":
        r = cls(s["k"], s["min_probability"])
        r.domains, r.W, r.b = s["domains"], s["W"].numpy(), s["b"].numpy()
        r.refs = {d: t.float().numpy() for d, t in s["refs"].items()}
        r.thresholds, r.domain_profiles, r.domain_titles = s["thresholds"], s["domain_profiles"], s["domain_titles"]
        return r


def gate(rec: dict, requested: str | None, covered: set[str]) -> tuple[str | None, dict]:
    """Routing/gating rule shared by the API service and the offline evaluation.

    Returns (profile_id to run, recognition dict with ``mode`` and optional ``block``/``note``).
    * auto mode (requested None/'auto'): run the recognised category's profile; unsupported -> block.
    * explicit profile covered by the recogniser: unsupported or a different recognised category -> block.
    * explicit profile not covered (custom product): no check, only a note.
    A ``block`` forces MANUAL INSPECTION REQUIRED in the decision engine.
    """
    rec = dict(rec)
    if requested in (None, "", "auto"):
        rec["mode"] = "auto"
        if not rec["supported"]:
            rec["block"] = ("UNSUPPORTED OR UNFAMILIAR COMPONENT: " + " ".join(rec["reasons"]) +
                            f" The '{rec['title']}' models were run for reference only.")
        return rec["profile_id"], rec
    rec["mode"] = "manual"
    if requested not in covered:
        rec["note"] = "The selected profile is not covered by the component recogniser (custom product); no category check."
    elif not rec["supported"]:
        rec["block"] = "UNSUPPORTED OR UNFAMILIAR COMPONENT: " + " ".join(rec["reasons"])
    elif rec["profile_id"] != requested:
        rec["block"] = (f"CATEGORY MISMATCH: '{requested}' was selected but the image looks like '{rec['title']}' "
                        f"(recogniser probability {rec['probability']:.2f}).")
    return requested, rec
