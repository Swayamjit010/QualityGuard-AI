"""Dataset discovery and layout recognition.

Supported layouts
-----------------
* ``folder_classification``: ``<root>/<class_name>/*.png`` (NEU-CLS, custom uploads)
* ``mvtec``: ``<root>/<category>/{train/good, test/<defect>, ground_truth/<defect>/*_mask.png}``
  (MVTec AD; also MPDD / BTAD that use the same convention)

Every discovered sample is returned as a plain dict so it can be serialized into a
versioned manifest without depending on torch.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ml.config import RAW_DIR, rel

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

# Human-curated metadata for the public datasets we ship download support for.
KNOWN_DATASETS = {
    "neu_cls": {
        "title": "NEU Surface Defect Database (NEU-CLS)",
        "task": "classification",
        "layout": "folder_classification",
        "license": "Research use; cite Song & Yan (2013), Northeastern University",
        "source": "http://faculty.neu.edu.cn/songkechen (mirror: huggingface.co/datasets/newguyme/neu_cls_caption)",
        "description": "Hot-rolled steel strip surface defects, 200x200 grayscale, 6 defect classes. "
                       "Contains NO defect-free class.",
    },
    "mvtec_ad": {
        "title": "MVTec Anomaly Detection (MVTec AD)",
        "task": "anomaly_detection",
        "layout": "mvtec",
        "license": "CC BY-NC-SA 4.0 (non-commercial)",
        "source": "https://www.mvtec.com/company/research/datasets/mvtec-ad (mirror: huggingface.co/datasets/foersben/mvtec-ad)",
        "description": "Industrial objects/textures. Train set = defect-free only; test set has "
                       "pixel-precise defect masks.",
    },
    "kolektor_sdd": {
        "title": "Kolektor Surface-Defect Dataset (KolektorSDD)",
        "task": "segmentation",
        "layout": "kolektor",
        "license": "CC BY-NC-SA 4.0",
        "source": "https://www.vicos.si/resources/kolektorsdd/",
        "description": "Electrical commutator surfaces with pixel-level crack annotations.",
    },
}

INVESTIGATED_DATASETS = [
    # Documented in docs/DATASETS.md; not downloaded by default because of size/licence/access.
    {"name": "MVTec LOCO AD", "reason": "Logical anomalies; ~3 GB; non-commercial licence"},
    {"name": "DAGM 2007", "reason": "Synthetic textures; requires registration on HCI Heidelberg site"},
    {"name": "KolektorSDD2", "reason": "~875 MB parquet mirror available; optional"},
    {"name": "BTAD", "reason": "MVTec-compatible layout; supported by the 'mvtec' loader"},
    {"name": "VisA", "reason": "~1.8 GB; exceeds disk budget of the reference machine"},
    {"name": "MPDD", "reason": "MVTec-compatible layout; supported by the 'mvtec' loader"},
]


@dataclass
class Sample:
    path: str
    label: str
    split_hint: str | None = None  # original split if the dataset defines one
    mask_path: str | None = None
    group: str | None = None
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


def _images(d: Path) -> list[Path]:
    return sorted(p for p in d.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def detect_layout(root: Path) -> str:
    root = Path(root)
    subdirs = [d for d in root.iterdir() if d.is_dir()]
    if any((d / "train" / "good").is_dir() for d in subdirs) or (root / "train" / "good").is_dir():
        return "mvtec"
    if subdirs and all(_images(d) for d in subdirs):
        return "folder_classification"
    if _images(root) and any(p.stem.endswith("_label") for p in _images(root)):
        return "kolektor"
    # KolektorSDD: kos01/Part0.jpg + Part0_label.bmp
    if subdirs and any(any(p.stem.endswith("_label") for p in _images(d)) for d in subdirs[:3]):
        return "kolektor"
    return "unknown"


def scan_folder_classification(root: Path) -> list[Sample]:
    out = []
    for cls_dir in sorted(d for d in Path(root).iterdir() if d.is_dir()):
        for p in _images(cls_dir):
            out.append(Sample(path=rel(p), label=cls_dir.name))
    return out


def scan_mvtec_category(cat_root: Path) -> list[Sample]:
    cat_root = Path(cat_root)
    out: list[Sample] = []
    for p in _images(cat_root / "train" / "good"):
        out.append(Sample(path=rel(p), label="good", split_hint="train"))
    test = cat_root / "test"
    for defect_dir in sorted(d for d in test.iterdir() if d.is_dir()):
        for p in _images(defect_dir):
            mask = None
            if defect_dir.name != "good":
                m = cat_root / "ground_truth" / defect_dir.name / f"{p.stem}_mask.png"
                mask = rel(m) if m.exists() else None
            out.append(Sample(path=rel(p), label=defect_dir.name, split_hint="test", mask_path=mask))
    return out


def scan_kolektor(root: Path) -> list[Sample]:
    out = []
    for p in sorted(Path(root).rglob("*")):
        if p.suffix.lower() in IMAGE_EXTS and not p.stem.endswith("_label"):
            m = next((q for q in p.parent.glob(f"{p.stem}_label.*")), None)
            out.append(Sample(path=rel(p), label="unknown", mask_path=rel(m) if m else None,
                              group=p.parent.name))  # group = physical commutator id
    return out


def scan(root: Path | str, layout: str | None = None) -> tuple[str, list[Sample]]:
    root = Path(root)
    layout = layout or detect_layout(root)
    if layout == "folder_classification":
        return layout, scan_folder_classification(root)
    if layout == "mvtec":
        if (root / "train" / "good").is_dir():
            return layout, scan_mvtec_category(root)
        samples = []
        for cat in sorted(d for d in root.iterdir() if d.is_dir() and (d / "train").is_dir()):
            for s in scan_mvtec_category(cat):
                s.meta["category"] = cat.name
                samples.append(s)
        return layout, samples
    if layout == "kolektor":
        return layout, scan_kolektor(root)
    raise ValueError(f"Unrecognized dataset layout at {root}")


def discover_local() -> list[dict]:
    """List datasets present under data/raw with their recognized layout."""
    found = []
    if not RAW_DIR.exists():
        return found
    for d in sorted(p for p in RAW_DIR.iterdir() if p.is_dir()):
        try:
            layout = detect_layout(d)
        except Exception:
            layout = "unknown"
        meta = KNOWN_DATASETS.get(d.name, {})
        found.append({"name": d.name, "path": rel(d), "layout": layout, **meta})
    return found
