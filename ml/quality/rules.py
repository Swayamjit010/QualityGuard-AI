"""Severity assessment and quality decision engine.

Three layers are kept explicitly separate in every result:

1. ``ai_assessment``  - what the models measured (class, anomaly score, regions, reliability)
2. ``rule_decision``  - what the *configured* product profile rules conclude from (1)
3. ``human_decision`` - filled in later by an inspector in the Review Center (never by this module)

Nothing here encodes a universal industrial tolerance; every threshold comes from the product profile.
The ``quality_index`` is a transparent, rule-derived deduction score (documented in
docs/QUALITY_RULES.md); it is NOT a scientifically validated measure of product quality.
"""
from __future__ import annotations

SEVERITY_ORDER = ["none", "low", "medium", "high", "critical"]
STATUSES = ("PASS", "FAIL", "REVIEW REQUIRED", "INCONCLUSIVE")
# User-facing inspection verdict (the primary output). ``status`` stays the profile's line disposition.
VERDICTS = ("NORMAL", "DEFECTIVE", "MANUAL INSPECTION REQUIRED")
VERDICT_TEXT = {
    "NORMAL": "No significant anomaly detected within the model's validated scope.",
    "DEFECTIVE": "A supported defect or significant anomaly has been detected.",
    "MANUAL INSPECTION REQUIRED": "The image is unfamiliar, ambiguous, out of distribution, unsuitable for inspection "
                                  "or the engines disagree, so no automatic verdict is given.",
}


def verdict_for(status: str, defect_present: bool | None) -> str:
    """Map the rule disposition to the verdict.

    REVIEW REQUIRED / INCONCLUSIVE -> MANUAL INSPECTION REQUIRED. FAIL -> DEFECTIVE. PASS -> NORMAL, unless a
    defect was detected but tolerated by the profile (severity below the reject level): the verdict is then still
    DEFECTIVE, because a defect *was* detected; the disposition (accept) is reported separately."""
    if status in ("REVIEW REQUIRED", "INCONCLUSIVE"):
        return "MANUAL INSPECTION REQUIRED"
    if status == "FAIL" or defect_present:
        return "DEFECTIVE"
    return "NORMAL"
UNFAMILIAR_MSG = "UNRECOGNIZED OR UNFAMILIAR PATTERN — MANUAL REVIEW REQUIRED"


def _sev_max(a: str, b: str) -> str:
    return a if SEVERITY_ORDER.index(a) >= SEVERITY_ORDER.index(b) else b


def _sev_shift(a: str, k: int) -> str:
    i = min(len(SEVERITY_ORDER) - 1, max(1, SEVERITY_ORDER.index(a) + k))
    return SEVERITY_ORDER[i]


def assess_severity(profile: dict, defect_present: bool, defect_class: str | None,
                    measurements: dict | None, anomaly_ratio: float | None) -> dict:
    """Return {'level', 'basis': [...explanations...], 'inputs': {...}}."""
    rules = profile.get("severity_rules", {})
    basis: list[str] = []
    if not defect_present:
        return {"level": "none", "basis": ["No defect detected by the enabled engines."], "inputs": {}}
    level = rules.get("default_level", "medium")
    basis.append(f"Profile default severity for a detected defect: {level}.")

    metric = rules.get("area_metric", "defect_area_ratio")
    if measurements and measurements.get("num_regions", 0) > 0 and rules.get("area_levels"):
        v = float(measurements.get(metric, 0.0))
        area_level = rules["area_levels"][-1]["level"]
        for band in rules["area_levels"]:
            if band.get("max") is None or v <= band["max"]:
                area_level = band["level"]
                break
        level = area_level
        basis.append(f"{metric}={v:.4%} -> '{area_level}' per configured area bands.")
        ce = rules.get("count_escalation")
        if ce and measurements["num_regions"] >= ce.get("regions_ge", 10 ** 9):
            level = _sev_shift(level, ce.get("escalate_by", 1))
            basis.append(f"{measurements['num_regions']} regions >= {ce['regions_ge']} -> escalated to '{level}'.")
        if measurements.get("calibrated") and rules.get("max_region_area_mm2") is not None:
            big = measurements["regions"][0].get("area_mm2", 0)
            if big > rules["max_region_area_mm2"]:
                level = _sev_max(level, "high")
                basis.append(f"Largest region {big:.2f} mm² > {rules['max_region_area_mm2']} mm² -> at least 'high'.")
    elif measurements is None and rules.get("area_levels"):
        basis.append("No localisation available; area-based bands not applied.")

    ie = rules.get("intensity_escalation")
    if ie and anomaly_ratio is not None and anomaly_ratio >= ie.get("score_ratio_ge", 1e9):
        level = _sev_shift(level, ie.get("escalate_by", 1))
        basis.append(f"Anomaly score is {anomaly_ratio:.2f}x threshold (>= {ie['score_ratio_ge']}) -> '{level}'.")

    cls_min = rules.get("class_min_level", {})
    if defect_class and defect_class in cls_min:
        new = _sev_max(level, cls_min[defect_class])
        if new != level:
            basis.append(f"Class '{defect_class}' has configured minimum severity '{cls_min[defect_class]}'.")
        level = new
    return {"level": level, "basis": basis,
            "inputs": {"defect_class": defect_class, "anomaly_ratio": anomaly_ratio,
                       "area_ratio": (measurements or {}).get("defect_area_ratio"),
                       "num_regions": (measurements or {}).get("num_regions")}}


def decide(profile: dict, *, input_ok: bool, input_issues: list[str], unfamiliar: bool,
           engines_disagree: bool, reliability: str, defect_present: bool | None,
           defect_class: str | None, severity: dict) -> dict:
    out = _decide(profile, input_ok=input_ok, input_issues=input_issues, unfamiliar=unfamiliar,
                  engines_disagree=engines_disagree, reliability=reliability, defect_present=defect_present,
                  defect_class=defect_class, severity=severity)
    out["verdict"] = verdict_for(out["status"], defect_present)
    out["verdict_text"] = VERDICT_TEXT[out["verdict"]]
    if out["verdict"] == "DEFECTIVE" and out["status"] == "PASS":
        out["verdict_text"] += " Its severity is below the profile's reject level, so the configured disposition is accept."
    return out


def _decide(profile: dict, *, input_ok: bool, input_issues: list[str], unfamiliar: bool,
           engines_disagree: bool, reliability: str, defect_present: bool | None,
           defect_class: str | None, severity: dict) -> dict:
    acc = profile.get("acceptance_rules", {})
    reasons: list[str] = []
    applied: list[str] = []
    if not input_ok:
        applied.append("input_validation")
        return _out("INCONCLUSIVE", ["Input image failed validation: " + "; ".join(input_issues)], applied,
                    severity, profile, review=True)
    if unfamiliar:
        applied.append("unfamiliar_pattern")
        return _out(acc.get("unfamiliar_action", "INCONCLUSIVE"), [UNFAMILIAR_MSG], applied, severity, profile,
                    review=True)
    if engines_disagree:
        applied.append("engine_disagreement")
        return _out(acc.get("disagreement_action", "REVIEW REQUIRED"),
                    ["Detection engines disagree or a score lies inside the configured review margin."],
                    applied, severity, profile, review=True)
    if reliability == "low" and acc.get("review_on_low_reliability", True):
        applied.append("low_reliability")
        return _out("REVIEW REQUIRED", ["Prediction reliability is LOW (see reliability checks)."], applied,
                    severity, profile, review=True)
    if not defect_present:
        applied.append("no_defect")
        return _out("PASS", ["No defect detected by the enabled engines."], applied, severity, profile)
    if defect_class and defect_class in acc.get("rejected_classes", []):
        applied.append("rejected_class")
        return _out("FAIL", [f"Defect class '{defect_class}' is configured as always-reject."], applied, severity,
                    profile)
    fail_at = acc.get("fail_at_severity", "medium")
    lvl = severity["level"]
    applied.append(f"fail_at_severity={fail_at}")
    if SEVERITY_ORDER.index(lvl) >= SEVERITY_ORDER.index(fail_at):
        reasons.append(f"Severity '{lvl}' >= configured reject level '{fail_at}'.")
        return _out("FAIL", reasons, applied, severity, profile)
    reasons.append(f"Defect present but severity '{lvl}' is below configured reject level '{fail_at}'.")
    status = "PASS" if acc.get("accept_below_fail_level", True) else "REVIEW REQUIRED"
    return _out(status, reasons, applied, severity, profile, review=status != "PASS")


def _out(status, reasons, applied, severity, profile, review=False) -> dict:
    ded = profile.get("quality_index", {}).get("deductions",
                                               {"none": 0, "low": 10, "medium": 35, "high": 65, "critical": 100})
    if status == "INCONCLUSIVE":
        qi = None
    else:
        qi = max(0, 100 - ded.get(severity.get("level", "none"), 0))
        if status == "REVIEW REQUIRED":
            qi = min(qi, profile.get("quality_index", {}).get("review_cap", 60))
    rec = {
        "PASS": "Accept. No action required beyond standard sampling.",
        "FAIL": "Reject / segregate the item and record the defect. Follow the profile's corrective-action guidance.",
        "REVIEW REQUIRED": "Route to a qualified inspector for manual confirmation before disposition.",
        "INCONCLUSIVE": "Do not disposition automatically. Re-capture the image or inspect manually.",
    }[status]
    return {"status": status, "reasons": reasons, "rules_applied": applied, "manual_review_required": review or
            status in ("REVIEW REQUIRED", "INCONCLUSIVE"), "quality_index": qi,
            "quality_index_note": "Rule-derived deduction index from the product profile; not a validated quality metric.",
            "recommendation": rec, "profile_id": profile.get("id"), "profile_version": profile.get("version", 1)}
