"""Hybrid inspection pipeline.

image -> validation -> preprocessing -> [Engine A classification ensemble] -> [Engine B anomaly
detection + localisation] -> [Engine C unfamiliar-pattern check] -> [Engine D fusion/reliability]
-> measurements -> severity -> configured quality rules -> visualisations

Models are lazy-loaded once and cached (thread-safe). Missing checkpoints raise
``ModelUnavailable`` - they are never silently replaced by a mock.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from ml.anomaly_detection.models import FeatureExtractor, GlobalNovelty, PaDiM, PatchCore
from ml.architectures.registry import EmbeddingHook, build_model
from ml.config import CHECKPOINT_DIR
from ml.explainability.cam import CAMExplainer, heatmap, overlay
from ml.inference.validation import validate_image
from ml.preprocessing.transforms import PreprocessConfig, preprocess_for_inference
from ml.quality.rules import UNFAMILIAR_MSG, assess_severity, decide
from ml.segmentation.regions import draw_regions, extract_regions
from ml.uncertainty.calibration import MahalanobisOOD, energy_score, softmax_np

DISPLAY_MAX = 512


class ModelUnavailable(RuntimeError):
    pass


class LoadedClassifier:
    def __init__(self, run_id: str):
        d = CHECKPOINT_DIR / run_id
        if not (d / "model.pt").exists():
            raise ModelUnavailable(f"Classifier checkpoint not found: {d / 'model.pt'}")
        t = time.perf_counter()
        ck = torch.load(d / "model.pt", map_location="cpu", weights_only=False)
        self.run_id, self.arch, self.classes = run_id, ck["arch"], ck["classes"]
        self.T = float(ck.get("temperature", 1.0))
        self.pp = PreprocessConfig.from_dict(ck["preprocess"])
        self.model = build_model(self.arch, len(self.classes), pretrained=False)
        self.model.load_state_dict(ck["state_dict"])
        self.model.eval()
        o = np.load(d / "ood.npz")
        self.ood = MahalanobisOOD.from_state({k: o[k] for k in o.files})
        self.ood.threshold, self.ood.energy_threshold = float(o["threshold"]), float(o["energy_threshold"])
        self.load_ms = (time.perf_counter() - t) * 1000
        self.lock = threading.Lock()

    def predict(self, x: torch.Tensor):
        with self.lock, torch.no_grad():
            hook = EmbeddingHook(self.model, self.arch)
            logits = self.model(x).numpy()
            emb = hook.value.numpy()
            hook.remove()
        return logits, emb

    def explain(self, x: torch.Tensor, methods: list[str], class_idx: int):
        out = {}
        with self.lock:
            ex = CAMExplainer(self.model, self.arch)
            try:
                for m in methods:
                    fn = {"gradcam": ex.gradcam, "gradcam_pp": ex.gradcam_pp, "scorecam": ex.scorecam}[m]
                    out[m] = fn(x, class_idx)[0]
            finally:
                ex.remove()
        return out


class LoadedAnomaly:
    def __init__(self, run_id: str):
        d = CHECKPOINT_DIR / run_id
        if not (d / "anomaly.pt").exists():
            raise ModelUnavailable(f"Anomaly checkpoint not found: {d / 'anomaly.pt'}")
        t = time.perf_counter()
        ck = torch.load(d / "anomaly.pt", map_location="cpu", weights_only=False)
        self.run_id, self.ck = run_id, ck
        self.pp = PreprocessConfig.from_dict(ck["preprocess"])
        ex = FeatureExtractor(ck["backbone"])
        self.engines = {"patchcore": PatchCore(ex).load(ck["patchcore"]), "padim": PaDiM(ex).load(ck["padim"])}
        # engines that failed validation-based selection at training time are not loaded/used for decisions
        if ck.get("deployed_engines"):
            self.engines = {k: v for k, v in self.engines.items() if k in ck["deployed_engines"]}
        self.novelty = GlobalNovelty(ex).load(ck["global_novelty"])
        self.thresholds = ck["thresholds"]
        self.pixel_thresholds = ck["pixel_thresholds"]
        self.margin = ck["review_margin"]
        # "all": every engine must agree (built-in MVTec models); "primary": the primary engine decides and the
        # others can only escalate to REVIEW (custom products, where PaDiM is weak with few training photos)
        self.fusion_mode = ck.get("fusion_mode", "all")
        self.primary_engine = ck.get("primary_engine")
        self.load_ms = (time.perf_counter() - t) * 1000
        self.lock = threading.Lock()


RECOGNIZER_RUN = "component_recognizer-resnet18-v1"


class LoadedRecognizer:
    def __init__(self, run_id: str = RECOGNIZER_RUN):
        from ml.recognition.router import ComponentRecognizer
        p = CHECKPOINT_DIR / run_id / "recognizer.pt"
        if not p.exists():
            raise ModelUnavailable(f"Component recogniser checkpoint not found: {p}. "
                                   "Train it with: python scripts/train_recognizer.py")
        t = time.perf_counter()
        self.run_id = run_id
        self.rec = ComponentRecognizer.load(torch.load(p, map_location="cpu", weights_only=False))
        self.rec._emb = __import__("ml.recognition.router", fromlist=["Embedder"]).Embedder()
        self.load_ms = (time.perf_counter() - t) * 1000

    def recognize(self, img_bgr) -> dict:
        r = self.rec.recognize(img_bgr)
        r["model"] = self.run_id
        return r


class LoadedSegmenter:
    def __init__(self, run_id: str):
        from ml.segmentation.unet import ResNetUNet
        p = CHECKPOINT_DIR / run_id / "segmentation.pt"
        if not p.exists():
            raise ModelUnavailable(f"Segmentation checkpoint not found: {p}")
        t = time.perf_counter()
        ck = torch.load(p, map_location="cpu", weights_only=False)
        self.run_id, self.pp = run_id, ck["preprocess"]
        self.model = ResNetUNet(pretrained=False)
        self.model.load_state_dict(ck["state_dict"])
        self.model.eval()
        self.image_threshold, self.pixel_threshold = float(ck["image_threshold"]), float(ck["pixel_threshold"])
        self.load_ms = (time.perf_counter() - t) * 1000
        self.lock = threading.Lock()

    def predict(self, img_bgr) -> tuple[float, np.ndarray]:
        from ml.segmentation.unet import image_score
        h, w = self.pp["height"], self.pp["width"]
        rgb = cv2.resize(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), (w, h), interpolation=cv2.INTER_AREA)
        x = (rgb.astype(np.float32) / 255 - np.array(self.pp["mean"], np.float32)) / np.array(self.pp["std"], np.float32)
        x = torch.from_numpy(x.transpose(2, 0, 1).copy())[None]
        with self.lock, torch.no_grad():
            p = torch.sigmoid(self.model(x))
            return float(image_score(p)[0]), p[0, 0].numpy()


class ModelCache:
    def __init__(self):
        self._c: dict[str, object] = {}
        self._lock = threading.Lock()

    def get(self, kind: str, run_id: str):
        key = f"{kind}:{run_id}"
        with self._lock:
            if key not in self._c:
                self._c[key] = {"classifier": LoadedClassifier, "anomaly": LoadedAnomaly, "recognizer": LoadedRecognizer,
                                "segmentation": LoadedSegmenter}[kind](run_id)
            return self._c[key]

    def loaded(self) -> list[dict]:
        return [{"key": k, "load_ms": getattr(v, "load_ms", None)} for k, v in self._c.items()]

    def evict(self, run_id: str):
        with self._lock:
            for k in [k for k in self._c if k.endswith(":" + run_id)]:
                del self._c[k]


CACHE = ModelCache()


def _display(rgb: np.ndarray) -> np.ndarray:
    h, w = rgb.shape[:2]
    s = DISPLAY_MAX / max(h, w)
    return cv2.resize(rgb, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else rgb


def _resize_map(m: np.ndarray, hw) -> np.ndarray:
    return cv2.resize(m.astype(np.float32), (hw[1], hw[0]), interpolation=cv2.INTER_LINEAR)


def reliability_level(conf: float, margin: float, agree: bool, ood: bool, prof: dict) -> tuple[str, list[str]]:
    r = prof.get("reliability", {})
    notes = []
    if ood:
        notes.append("Embedding is outside the training distribution.")
    if not agree:
        notes.append("Ensemble members disagree on the top class.")
    if conf < r.get("min_confidence", 0.6):
        notes.append(f"Calibrated confidence {conf:.2f} < {r.get('min_confidence', 0.6)}.")
    if margin < r.get("min_margin", 0.2):
        notes.append(f"Top-1/top-2 probability margin {margin:.2f} < {r.get('min_margin', 0.2)}.")
    if ood or not agree or conf < r.get("min_confidence", 0.6):
        return "low", notes
    if margin < r.get("min_margin", 0.2) or conf < r.get("high_confidence", 0.9):
        return "medium", notes
    return "high", notes or ["Confident, consistent and in-distribution."]


def inspect(img_bgr: np.ndarray | None, profile: dict, options: dict | None = None) -> dict:
    """Run the full hybrid pipeline. Returns a JSON-serialisable result plus ``_images`` (RGB arrays)
    and ``_embedding`` (np.ndarray) for the caller to persist."""
    options = options or {}
    t_all = time.perf_counter()
    timings: dict[str, float] = {}
    ok, issues, info = validate_image(img_bgr, profile)
    result: dict = {"profile_id": profile["id"], "input": info, "input_valid": ok, "input_issues": issues,
                    "engines": {}, "_images": {}, "_embedding": None}
    if options.get("recognition"):
        result["recognition"] = options["recognition"]
    if not ok:
        result["severity"] = {"level": "none", "basis": [], "inputs": {}}
        result["decision"] = decide(profile, input_ok=False, input_issues=issues, unfamiliar=False,
                                    engines_disagree=False, reliability="low", defect_present=None,
                                    defect_class=None, severity=result["severity"])
        result["timings_ms"] = {"total": (time.perf_counter() - t_all) * 1000}
        return result

    oh, ow = img_bgr.shape[:2]
    disp = _display(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    dh, dw = disp.shape[:2]
    result["_images"]["original"] = disp
    models_cfg = profile.get("models", {})
    enabled = options.get("engines") or profile.get("pipeline", {}).get("engines",
                                                                        ["classification", "anomaly", "novelty"])
    cam_methods = options.get("cam_methods") or profile.get("pipeline", {}).get("cam_methods", ["gradcam"])
    defect_present, defect_class, measurements, anomaly_ratio = None, None, None, None
    anomaly_disagree = False
    unfamiliar, disagree, reliability, rel_notes = False, False, "high", []
    used_models = []

    # ---------------- Engine A: supervised classification (ensemble)
    clf_ids = options.get("classifiers") or models_cfg.get("classifiers", [])
    if "classification" in enabled and clf_ids:
        t = time.perf_counter()
        members, probs_all = [], []
        for rid in clf_ids:
            c = CACHE.get("classifier", rid)
            x, rgb_model = preprocess_for_inference(img_bgr, c.pp)
            lg, emb = c.predict(x[None])
            p = softmax_np(lg, c.T)[0]
            maha = float(c.ood.score(emb)[0])
            en = float(energy_score(lg, c.T)[0])
            members.append({"run_id": rid, "arch": c.arch, "classes": c.classes,
                            "probabilities": {k: float(v) for k, v in zip(c.classes, p)},
                            "top_class": c.classes[int(p.argmax())], "confidence": float(p.max()),
                            "mahalanobis": maha, "mahalanobis_threshold": c.ood.threshold,
                            "energy": en, "energy_threshold": c.ood.energy_threshold,
                            "ood": bool(maha > c.ood.threshold), "temperature": c.T})
            probs_all.append(p)
            used_models.append(rid)
            if result["_embedding"] is None:
                result["_embedding"] = emb[0].astype(np.float32)
                result["embedding_model"] = rid
        classes = members[0]["classes"]
        mean_p = np.mean(probs_all, 0)  # same label space + each member temperature-calibrated
        order = np.argsort(-mean_p)
        top, conf = classes[int(order[0])], float(mean_p[order[0]])
        margin = float(mean_p[order[0]] - mean_p[order[1]]) if len(order) > 1 else 1.0
        agree = len({m["top_class"] for m in members}) == 1
        ood_votes = sum(m["ood"] for m in members)
        clf_ood = ood_votes > len(members) / 2
        reliability, rel_notes = reliability_level(conf, margin, agree, clf_ood, profile)
        defect_class = top if top not in profile.get("normal_classes", ["good", "normal"]) else None
        if profile.get("assumes_defect_present"):
            defect_present = True
        else:
            defect_present = defect_class is not None
        disagree = disagree or not agree
        unfamiliar = unfamiliar or clf_ood
        result["engines"]["classification"] = {
            "ensemble": members, "fusion": "mean of temperature-calibrated probabilities (same label space)",
            "probabilities": {k: float(v) for k, v in zip(classes, mean_p)}, "predicted_class": top,
            "confidence": conf, "margin": margin, "members_agree": agree, "ood_votes": ood_votes,
            "unfamiliar": clf_ood, "reliability": reliability, "reliability_notes": rel_notes,
            "entropy": float(-(mean_p * np.log(mean_p + 1e-12)).sum()),
        }
        timings["classification"] = (time.perf_counter() - t) * 1000

        # ---------------- Explainability (primary model)
        if cam_methods and not options.get("skip_explain"):
            t = time.perf_counter()
            c = CACHE.get("classifier", clf_ids[0])
            x, rgb_model = preprocess_for_inference(img_bgr, c.pp)
            cams = c.explain(x[None], cam_methods, c.classes.index(top))
            for m, cam in cams.items():
                cam_d = _resize_map(cam, (dh, dw))
                result["_images"][f"{m}_overlay"] = overlay(disp, cam_d)
                result["_images"][f"{m}_heatmap"] = heatmap(cam_d)
                result["engines"]["classification"].setdefault("explanations", {})[m] = {
                    "target_class": top, "model": clf_ids[0], "peak_location_norm": [
                        float(np.unravel_index(cam.argmax(), cam.shape)[1] / cam.shape[1]),
                        float(np.unravel_index(cam.argmax(), cam.shape)[0] / cam.shape[0])],
                    "activation_area_above_0.5": float((cam > 0.5).mean())}
            timings["explainability"] = (time.perf_counter() - t) * 1000

    # ---------------- Engine B: anomaly detection + localisation
    an_id = options.get("anomaly_model") or models_cfg.get("anomaly")
    if "anomaly" in enabled and an_id:
        t = time.perf_counter()
        a = CACHE.get("anomaly", an_id)
        used_models.append(an_id)
        x, rgb_model = preprocess_for_inference(img_bgr, a.pp)
        scores, maps, votes, near = {}, {}, {}, False
        with a.lock:
            for name, eng in a.engines.items():
                s, m = eng.predict(x[None])
                scores[name], maps[name] = float(s[0]), m[0]
                votes[name] = scores[name] >= a.thresholds[name]
                near = near or abs(scores[name] / a.thresholds[name] - 1) < a.margin
            nov = float(a.novelty.score(x[None])[0])
        an_unfamiliar = nov > a.novelty.threshold
        primary = a.primary_engine or profile.get("pipeline", {}).get("primary_anomaly_engine", "patchcore")
        if primary not in a.engines:
            primary = next(iter(a.engines))
        anomaly_ratio = scores[primary] / a.thresholds[primary]
        if a.fusion_mode == "primary":
            near = abs(anomaly_ratio - 1) < a.margin
            is_anom = votes[primary] and not near
            # a secondary engine can only escalate a PASS to REVIEW, never turn a FAIL into a PASS
            an_disagree = near or (not votes[primary] and any(v for k, v in votes.items() if k != primary))
        else:
            all_yes, all_no = all(votes.values()), not any(votes.values())
            an_disagree = (not all_yes and not all_no) or near
            is_anom = all_yes
        px_t = a.pixel_thresholds[primary]
        mask = maps[primary] >= px_t
        cal = profile.get("calibration", {}).get("mm_per_pixel")
        if is_anom or (an_disagree and votes.get(primary)):
            measurements = extract_regions(mask, maps[primary], profile.get("min_region_area_px", 30), (oh, ow), cal)
        # visualisations from the actual maps
        for name, m in maps.items():
            norm = np.clip(_resize_map(m, (dh, dw)) / (2 * a.pixel_thresholds[name]), 0, 1)
            result["_images"][f"anomaly_{name}_heatmap"] = heatmap(norm)
            result["_images"][f"anomaly_{name}_overlay"] = overlay(disp, norm)
        mask_d = _resize_map(mask.astype(np.float32), (dh, dw)) > 0.5
        seg = disp.copy()
        seg[mask_d] = (0.4 * seg[mask_d] + 0.6 * np.array([255, 40, 40])).astype(np.uint8)
        cnts, _ = cv2.findContours(mask_d.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(seg, cnts, -1, (255, 255, 0), 2)
        result["_images"]["segmentation_mask"] = (mask_d * 255).astype(np.uint8)
        result["_images"]["segmentation_overlay"] = seg
        if measurements:
            sx, sy = dw / ow, dh / oh
            scaled = {**measurements, "regions": [{**r, "bbox_px": [int(r["bbox_px"][0] * sx), int(r["bbox_px"][1] * sy),
                                                                   int(r["bbox_px"][2] * sx), int(r["bbox_px"][3] * sy)]}
                                                  for r in measurements["regions"]]}
            result["_images"]["bounding_boxes"] = draw_regions(disp, scaled)
            if measurements["regions"]:
                x0, y0, bw, bh = scaled["regions"][0]["bbox_px"]
                pad = int(max(bw, bh) * 0.5) + 8
                crop = disp[max(0, y0 - pad):y0 + bh + pad, max(0, x0 - pad):x0 + bw + pad]
                if crop.size:
                    result["_images"]["defect_zoom"] = cv2.resize(crop, (256, int(256 * crop.shape[0] / crop.shape[1])),
                                                                  interpolation=cv2.INTER_CUBIC)
        result["engines"]["anomaly"] = {
            "model": an_id, "scores": scores, "thresholds": a.thresholds, "votes": votes,
            "score_ratio": {k: scores[k] / a.thresholds[k] for k in scores}, "primary_engine": primary,
            "is_anomalous": is_anom, "engines_disagree": an_disagree, "within_review_margin": near,
            "review_margin": a.margin, "pixel_threshold": px_t, "defect_pixels_fraction": float(mask.mean()),
            "fusion": ("primary engine decides (FAIL above its threshold); within the review margin, or another engine "
                       "flagging a primary PASS -> REVIEW") if a.fusion_mode == "primary" else
                      "FAIL if all engines exceed own validated threshold; disagreement or margin -> REVIEW",
        }
        result["engines"]["novelty"] = {"score": nov, "threshold": a.novelty.threshold, "unfamiliar": an_unfamiliar,
                                        "method": "global ResNet feature kNN distance to defect-free references"}
        if result["_embedding"] is None:
            with a.lock, torch.no_grad():
                result["_embedding"] = a.novelty._g(x[None])[0].numpy().astype(np.float32)
            result["embedding_model"] = an_id
        unfamiliar = unfamiliar or (an_unfamiliar and "novelty" in enabled)
        anomaly_disagree = an_disagree  # applied below only if the anomaly engine votes
        if defect_present is None or not profile.get("assumes_defect_present"):
            defect_present = bool(is_anom) if defect_present is None else (defect_present or bool(is_anom))
        timings["anomaly"] = (time.perf_counter() - t) * 1000
        if "classification" not in result["engines"]:
            reliability = "high" if not (an_disagree or an_unfamiliar) else "low"
            rel_notes = []
            if near:
                rel_notes.append("A score lies within the review margin of its threshold.")
            if an_unfamiliar:
                rel_notes.append("Global appearance differs from all defect-free references.")
            if an_disagree and not near:
                rel_notes.append("PatchCore and PaDiM disagree.")
            if not rel_notes:
                rel_notes = ["Both anomaly engines agree and the score is outside the review margin."]

    # ---------------- Engine S: supervised segmentation (pixel-annotated training data)
    seg_id = options.get("segmentation_model") or models_cfg.get("segmentation")
    seg_vote = None
    if "segmentation" in enabled and seg_id:
        t = time.perf_counter()
        sg = CACHE.get("segmentation", seg_id)
        used_models.append(seg_id)
        s_score, prob = sg.predict(img_bgr)
        seg_vote = s_score >= sg.image_threshold
        seg_margin = profile.get("pipeline", {}).get("segmentation_review_margin", 0.0)
        seg_near = seg_margin > 0 and abs(s_score - sg.image_threshold) < seg_margin  # absolute, in probability units
        prob_o = _resize_map(prob, (oh, ow))
        prob_d = _resize_map(prob, (dh, dw))
        mask_o = prob_o >= sg.pixel_threshold
        mask_d = prob_d >= sg.pixel_threshold
        result["_images"]["segmentation_unet_heatmap"] = heatmap(np.clip(prob_d, 0, 1))
        result["_images"]["segmentation_unet_overlay"] = overlay(disp, np.clip(prob_d, 0, 1))
        seg = disp.copy()
        seg[mask_d] = (0.4 * seg[mask_d] + 0.6 * np.array([255, 40, 40])).astype(np.uint8)
        cnts, _ = cv2.findContours(mask_d.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(seg, cnts, -1, (255, 255, 0), 2)
        # the supervised mask is the precise boundary estimate: it replaces the anomaly-map mask in the UI
        result["_images"]["segmentation_mask"] = (mask_d * 255).astype(np.uint8)
        result["_images"]["segmentation_overlay"] = seg
        if seg_vote:
            cal = profile.get("calibration", {}).get("mm_per_pixel")
            measurements = extract_regions(mask_o, prob_o, profile.get("min_region_area_px", 30), (oh, ow), cal)
            if measurements and measurements["regions"]:
                sx, sy = dw / ow, dh / oh
                scaled = {**measurements, "regions": [{**r, "bbox_px": [int(r["bbox_px"][0] * sx), int(r["bbox_px"][1] * sy),
                                                                       int(r["bbox_px"][2] * sx), int(r["bbox_px"][3] * sy)]}
                                                      for r in measurements["regions"]]}
                result["_images"]["bounding_boxes"] = draw_regions(disp, scaled)
                x0, y0, bw, bh = scaled["regions"][0]["bbox_px"]
                pad = int(max(bw, bh) * 0.5) + 8
                crop = disp[max(0, y0 - pad):y0 + bh + pad, max(0, x0 - pad):x0 + bw + pad]
                if crop.size:
                    result["_images"]["defect_zoom"] = cv2.resize(crop, (256, int(256 * crop.shape[0] / crop.shape[1])),
                                                                  interpolation=cv2.INTER_CUBIC)
        result["engines"]["segmentation"] = {
            "model": seg_id, "image_score": s_score, "image_threshold": sg.image_threshold,
            "score_ratio": s_score / sg.image_threshold, "is_defective": bool(seg_vote),
            "within_review_margin": bool(seg_near), "review_margin": seg_margin,
            "pixel_threshold": sg.pixel_threshold, "defect_pixels_fraction": float(mask_o.mean()),
            "method": "supervised U-Net (ResNet-18 encoder) trained on pixel-annotated defect masks",
        }
        if seg_near:
            disagree = True
            rel_notes.append("Segmentation score lies within the review margin of its validated threshold.")
        timings["segmentation"] = (time.perf_counter() - t) * 1000

    if not result["engines"]:
        raise ModelUnavailable("No inspection engine is configured/enabled for this profile.")

    # ---------------- cross-engine fusion (profiles that run several defect engines)
    votes = {}
    if "classification" in result["engines"] and not profile.get("assumes_defect_present"):
        votes["classification"] = defect_class is not None
    if "anomaly" in result["engines"]:
        votes["anomaly"] = bool(result["engines"]["anomaly"]["is_anomalous"])
    if seg_vote is not None:
        votes["segmentation"] = bool(seg_vote)
    voting = profile.get("pipeline", {}).get("voting_engines")
    if voting and votes:
        # engines outside the validated voting set are still run and displayed, but do NOT influence the decision
        result["non_voting_engines"] = {k: v for k, v in votes.items() if k not in voting}
        votes = {k: v for k, v in votes.items() if k in voting}
    if anomaly_disagree and (not voting or "anomaly" in voting):
        disagree = True
    if votes and (voting or len(votes) > 1):
        decision_engine = profile.get("pipeline", {}).get("decision_engine")
        agree = len(set(votes.values())) == 1
        if decision_engine in votes:
            defect_present = votes[decision_engine]
            rule = f"'{decision_engine}' decides; any other voting engine disagreeing -> manual inspection"
        else:
            defect_present = next(iter(votes.values())) if agree else bool(defect_present)
            rule = "all voting engines must agree; any disagreement -> manual inspection"
        if not agree:
            disagree = True
            rel_notes.append("Engines disagree: " + ", ".join(f"{k}={'defect' if v else 'no defect'}"
                                                              for k, v in votes.items()) + ".")
        result["fusion"] = {"votes": votes, "rule": rule, "disagree": not agree,
                            "non_voting": result.get("non_voting_engines", {})}
    elif seg_vote is not None and defect_present is None:
        defect_present = bool(seg_vote)

    # ---------------- component recognition gate (computed by the caller, see inspection_service)
    recog = options.get("recognition")
    if recog:
        if recog.get("block"):
            unfamiliar = True
            rel_notes.append(recog["block"])
            # the engines of the closest category ran for reference only: their findings are not recorded as a result
            result["reference_only_findings"] = {"defect_present": defect_present, "defect_class": defect_class}
            defect_present, defect_class, measurements = False, None, None

    severity = assess_severity(profile, bool(defect_present), defect_class, measurements, anomaly_ratio)
    decision = decide(profile, input_ok=True, input_issues=[], unfamiliar=unfamiliar, engines_disagree=disagree,
                      reliability=reliability, defect_present=bool(defect_present), defect_class=defect_class,
                      severity=severity)
    if unfamiliar:
        decision["message"] = UNFAMILIAR_MSG
        if recog and recog.get("block"):
            decision["reasons"].append(recog["block"])
    result.update({
        "defect_present": bool(defect_present), "defect_class": defect_class, "measurements": measurements,
        "severity": severity, "decision": decision, "reliability": {"level": reliability, "notes": rel_notes},
        "unfamiliar": unfamiliar, "models_used": used_models,
    })
    timings["total"] = (time.perf_counter() - t_all) * 1000
    result["timings_ms"] = timings
    # comparison panel (original | primary explanation/anomaly)
    second = result["_images"].get("segmentation_overlay")
    if second is None and cam_methods:
        second = result["_images"].get(f"{cam_methods[0]}_overlay")
    if second is not None and second.shape == disp.shape:
        result["_images"]["comparison"] = np.concatenate([disp, np.full((dh, 6, 3), 255, np.uint8), second], 1)
    return result
