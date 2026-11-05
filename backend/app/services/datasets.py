"""Dataset management: inventory, ZIP/folder import, analysis, versioned splitting, sample browsing, export."""
from __future__ import annotations

import csv
import io
import json
import re
import shutil
import zipfile
from pathlib import Path

from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.models.entities import Dataset, DatasetVersion, HumanFeedback, ImageAnnotation, Inspection, utcnow
from app.services.storage import safe_name, storage
from ml.config import METRICS_DIR, PROCESSED_DIR, RAW_DIR, rel
from ml.datasets.analysis import analyze_samples
from ml.datasets.registry import IMAGE_EXTS, KNOWN_DATASETS, detect_layout, discover_local, scan
from ml.datasets.splitting import build_groups, stratified_group_split, write_manifest

NAME_RE = re.compile(r"^[a-z0-9_]{2,64}$")
MAX_ZIP_FILES = 50_000
MAX_ZIP_UNCOMPRESSED = 4 * 1024 ** 3


def sync(db: Session) -> None:
    for d in discover_local():
        ds = db.query(Dataset).filter_by(name=d["name"]).first()
        if not ds:
            ds = Dataset(name=d["name"], title=d.get("title", d["name"]), task=d.get("task", "classification"),
                         layout=d["layout"], source=d.get("source", "local import"), license=d.get("license", "unknown"),
                         path=d["path"], description=d.get("description", ""))
            db.add(ds)
            db.flush()
    # manifests -> versions (MVTec categories are registered as separate datasets mvtec_<cat>)
    for mdir in sorted(p for p in PROCESSED_DIR.iterdir() if p.is_dir()):
        latest = mdir / "manifest_latest.json"
        if not latest.exists():
            continue
        name = mdir.name
        ds = db.query(Dataset).filter_by(name=name).first()
        if not ds:
            base = KNOWN_DATASETS.get("mvtec_ad" if name.startswith("mvtec_") else name, {})
            ds = Dataset(name=name, title=(base.get("title", name) + (f" — {name[6:]}" if name.startswith("mvtec_") else "")),
                         task=base.get("task", "classification"), layout=base.get("layout", "folder_classification"),
                         source=base.get("source", "local"), license=base.get("license", "unknown"),
                         path=rel(RAW_DIR / ("mvtec_ad/" + name[6:] if name.startswith("mvtec_") else name)),
                         description=base.get("description", ""))
            db.add(ds)
            db.flush()
        for mf in sorted(mdir.glob("manifest_*.json")):
            if mf.name == "manifest_latest.json":
                continue
            doc = json.loads(mf.read_text())
            if db.query(DatasetVersion).filter_by(dataset_id=ds.id, version=doc["version"]).first():
                continue
            rep_p = METRICS_DIR / f"dataset_report_{name}.json"
            rep = json.loads(rep_p.read_text()) if rep_p.exists() else {}
            rep.pop("mean_histogram", None)
            db.add(DatasetVersion(dataset_id=ds.id, version=doc["version"], manifest_path=rel(mf),
                                  num_samples=doc["num_samples"], split_counts=doc["split_counts"],
                                  statistics=rep, split_policy=doc["meta"].get("split_policy", "")))
    db.commit()


def import_zip(db: Session, name: str, data: bytes, task: str = "classification") -> Dataset:
    if not NAME_RE.match(name):
        raise HTTPException(422, "Dataset name must be 2-64 chars of a-z, 0-9, _")
    if db.query(Dataset).filter_by(name=name).first() or (RAW_DIR / name).exists():
        raise HTTPException(409, "Dataset already exists")
    dest = RAW_DIR / name
    try:
        z = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile:
        raise HTTPException(422, "Not a valid ZIP archive")
    infos = z.infolist()
    if len(infos) > MAX_ZIP_FILES or sum(i.file_size for i in infos) > MAX_ZIP_UNCOMPRESSED:
        raise HTTPException(413, "ZIP too large (zip-bomb protection)")
    dest.mkdir(parents=True)
    n_img = 0
    try:
        for i in infos:
            if i.is_dir():
                continue
            parts = [safe_name(p) for p in Path(i.filename).parts if p not in ("..", ".", "")]
            if not parts or parts[0] == "__MACOSX":
                continue
            ext = Path(parts[-1]).suffix.lower()
            if ext not in IMAGE_EXTS | {".csv", ".json"}:
                continue
            target = (dest / Path(*parts)).resolve()
            if dest.resolve() not in target.parents:
                raise HTTPException(422, f"Unsafe path in archive: {i.filename}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(i) as src, open(target, "wb") as out:
                shutil.copyfileobj(src, out)
            n_img += ext in IMAGE_EXTS
        # unwrap single top-level folder
        kids = [p for p in dest.iterdir()]
        if len(kids) == 1 and kids[0].is_dir():
            inner = kids[0]
            for p in inner.iterdir():
                shutil.move(str(p), dest / p.name)
            inner.rmdir()
        _apply_csv_labels(dest)
        layout = detect_layout(dest)
        if layout == "unknown" or n_img == 0:
            raise HTTPException(422, "Unrecognised layout. Use <class>/<images> folders, MVTec layout, or images + "
                                     "labels.csv (filename,label).")
    except HTTPException:
        shutil.rmtree(dest, ignore_errors=True)
        raise
    ds = Dataset(name=name, title=name, task=task, layout=layout, source="user upload", license="user-provided",
                 path=rel(dest), description=f"Imported ZIP with {n_img} images")
    db.add(ds)
    db.commit()
    return ds


def _apply_csv_labels(root: Path):
    """Support flat folders + labels.csv (filename,label) or labels.json ({filename: label})."""
    csvp, jsp = root / "labels.csv", root / "labels.json"
    mapping = {}
    if csvp.exists():
        with open(csvp, newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                fn = row.get("filename") or row.get("file") or row.get("image")
                lb = row.get("label") or row.get("class")
                if fn and lb:
                    mapping[Path(fn).name] = safe_name(lb)
    elif jsp.exists():
        mapping = {Path(k).name: safe_name(str(v)) for k, v in json.loads(jsp.read_text()).items()}
    for fn, lb in mapping.items():
        src = root / fn
        if src.exists():
            (root / lb).mkdir(exist_ok=True)
            shutil.move(str(src), root / lb / fn)


def analyze_and_split(db: Session, ds: Dataset, ratios=(0.7, 0.15, 0.15), seed: int = 42) -> DatasetVersion:
    root = Path(ds.path) if Path(ds.path).is_absolute() else Path(__file__).resolve().parents[3] / ds.path
    layout, samples = scan(root, ds.layout if ds.layout != "unknown" else None)
    rows = [s.to_dict() for s in samples]
    if not rows:
        raise HTTPException(422, "Dataset contains no images")
    rep = analyze_samples(rows)
    bad = {c["path"] for c in rep["corrupted"]}
    rows = [r for r in rows if r["path"] not in bad]
    groups = build_groups(rows, rep["exact_duplicates"] + [list(x) for x in rep["near_duplicates"]])
    split = stratified_group_split(rows, groups, ratios, seed)
    for r, g, s in zip(rows, groups, split):
        r["group"], r["split"] = g, s
    mf = write_manifest(ds.name, rows, {"task": ds.task, "split_policy": "stratified; duplicates grouped",
                                        "seed": seed, "ratios": list(ratios)})
    rep = {k: v for k, v in rep.items() if not k.startswith("_")}
    rep["manifest"] = mf.name
    (METRICS_DIR / f"dataset_report_{ds.name}.json").write_text(json.dumps(rep, indent=2))
    doc = json.loads(mf.read_text())
    v = db.query(DatasetVersion).filter_by(dataset_id=ds.id, version=doc["version"]).first()
    if not v:
        rep.pop("mean_histogram", None)
        v = DatasetVersion(dataset_id=ds.id, version=doc["version"], manifest_path=rel(mf), num_samples=len(rows),
                           split_counts=doc["split_counts"], statistics=rep, split_policy=doc["meta"]["split_policy"])
        db.add(v)
        db.commit()
    return v


def export_feedback(db: Session, only_new: bool = True) -> dict:
    """Curated dataset from VERIFIED inspector feedback -> data/feedback/<ts>/<label>/<image> + manifest.csv."""
    from ml.config import DATA_DIR
    q = db.query(HumanFeedback)
    if only_new:
        q = q.filter(HumanFeedback.exported.is_(False))
    fbs = q.all()
    if not fbs:
        return {"exported": 0, "path": None}
    out = DATA_DIR / "feedback" / utcnow().strftime("%Y%m%d-%H%M%S")
    rows = []
    for f in fbs:
        insp = db.get(Inspection, f.inspection_id)
        if not insp:
            continue
        label = f.corrected_class or (insp.defect_class if f.verdict == "confirm" else None)
        if label is None:
            label = "good" if f.final_decision == "accept" else "defect_unspecified"
        src = next(storage.resolve(insp.artifact_dir).glob("upload.*"), None)
        if not src:
            continue
        d = out / insp.profile_id / safe_name(label)
        d.mkdir(parents=True, exist_ok=True)
        dst = d / f"{insp.id}{src.suffix}"
        shutil.copy2(src, dst)
        # verified localisation from the inspector travels with the image (masks + boxes)
        mask_rel, boxes = "", []
        anns = db.query(ImageAnnotation).filter_by(image_id=insp.image_id, source="inspector").all()             if insp.image_id else []
        for a in anns:
            if a.kind == "mask" and a.mask_path and not mask_rel:
                mdst = out / insp.profile_id / "masks" / f"{insp.id}.png"
                mdst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(storage.resolve(a.mask_path), mdst)
                mask_rel = rel(mdst)
            elif a.kind == "bbox":
                boxes += a.data.get("boxes", [])
        rows.append({"file": rel(dst), "profile_id": insp.profile_id, "label": label, "model_prediction": insp.defect_class,
                     "verdict": f.verdict, "final_decision": f.final_decision, "difficult": f.difficult,
                     "location_confirmed": f.location_confirmed, "mask": mask_rel,
                     "boxes": json.dumps(boxes) if boxes else "", "inspection_id": insp.id,
                     "reviewer": f.reviewer.email})
        f.exported = True
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "manifest.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]) if rows else ["file"])
        w.writeheader(); w.writerows(rows)
    db.commit()
    return {"exported": len(rows), "path": rel(out)}
