"""Background job execution (thread pool) with DB-persisted progress.

* batch / folder inspection
* video inspection (frame sampling + annotated WebM/VP8 output)
* production-line simulation (clearly flagged ``is_simulated``)
* training jobs run as **subprocesses** (isolation, stoppable via STOP file, resumable)

A Redis/RQ or Celery backend can replace ``JobManager`` without changing the API layer.
"""
from __future__ import annotations

import json
import logging
import os
import random
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from app.core.config import ROOT, get_settings
from app.database.session import session_scope
from app.models.entities import Job, ProductionBatch, ProductionLine, TrainingJob, User, utcnow
from app.services.events import bus
from app.services.inspection_service import run_inspection, summary
from app.services.storage import storage
from ml.config import CHECKPOINT_DIR, absp
from ml.datasets.splitting import load_manifest

log = logging.getLogger("qualityguard.jobs")
settings = get_settings()


class JobManager:
    def __init__(self, workers: int):
        self.pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="qg-job")
        self.stop_flags: dict[str, threading.Event] = {}
        self.procs: dict[str, subprocess.Popen] = {}

    def create(self, kind: str, params: dict, user: User | None) -> str:
        with session_scope() as db:
            j = Job(kind=kind, params=params, created_by=user.id if user else None)
            db.add(j)
            db.flush()
            return j.id

    def update(self, job_id: str, **kw):
        with session_scope() as db:
            j = db.get(Job, job_id)
            for k, v in kw.items():
                setattr(j, k, v)
        bus.publish("job", {"id": job_id, **{k: v for k, v in kw.items() if k in ("status", "progress", "message")}})

    def submit(self, job_id: str, fn, *args):
        self.stop_flags[job_id] = threading.Event()

        def run():
            self.update(job_id, status="running")
            try:
                res = fn(job_id, self.stop_flags[job_id], *args)
                st = "stopped" if self.stop_flags[job_id].is_set() else "completed"
                self.update(job_id, status=st, progress=1.0, result=res or {}, message=st)
            except Exception as exc:
                log.exception("job %s failed", job_id)
                self.update(job_id, status="failed", message=str(exc)[:2000])
        self.pool.submit(run)

    def stop(self, job_id: str) -> bool:
        ev = self.stop_flags.get(job_id)
        if ev:
            ev.set()
            return True
        return False


MANAGER = JobManager(settings.workers)


# ------------------------------------------------------------------ batch / folder
def batch_job(job_id: str, stop: threading.Event, files: list[tuple[str, str]], profile_id: str, batch_id: str,
              user_id: int | None, source: str, options: dict):
    """files: list of (storage_key_of_upload, original_name). Uploads are staged to storage first."""
    results, errors = [], []
    for i, (key, name) in enumerate(files):
        if stop.is_set():
            break
        try:
            data = storage.resolve(key).read_bytes()
            with session_scope() as db:
                user = db.get(User, user_id) if user_id else None
                insp = run_inspection(db, data=data, filename=name, profile_id=profile_id, user=user, source=source,
                                      batch_id=batch_id, options=options)
                results.append(summary(insp))
        except Exception as exc:
            errors.append({"file": name, "error": str(exc)[:300]})
        finally:
            storage.resolve(key).unlink(missing_ok=True)
        MANAGER.update(job_id, progress=(i + 1) / len(files), message=f"{i + 1}/{len(files)} processed")
    agg = _aggregate(results)
    with session_scope() as db:
        b = db.get(ProductionBatch, batch_id)
        b.summary = {**agg, "errors": errors}
    return {"batch_id": batch_id, "processed": len(results), "errors": errors, "summary": agg}


def _aggregate(results: list[dict]) -> dict:
    from collections import Counter
    st = Counter(r["status"] for r in results)
    return {"total": len(results), "status": dict(st),
            "defects": dict(Counter(r["defect_class"] or ("anomaly" if r["defect_present"] else None)
                                    for r in results if r["defect_present"])),
            "acceptance_rate": st.get("PASS", 0) / len(results) if results else None,
            "avg_latency_ms": float(np.mean([r["latency_ms"] for r in results])) if results else None}


# ------------------------------------------------------------------ video
STATUS_BGR = {"PASS": (60, 170, 30), "FAIL": (40, 40, 220), "REVIEW REQUIRED": (20, 140, 230),
              "INCONCLUSIVE": (120, 120, 120)}


def video_job(job_id: str, stop: threading.Event, video_key: str, profile_id: str, batch_id: str, user_id: int | None,
              every_s: float, max_frames: int, options: dict):
    path = storage.resolve(video_key)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError("Video could not be opened (unsupported codec/container)")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, int(round(every_s * fps)))
    targets = list(range(0, n if n > 0 else 10 ** 9, step))[:max_frames]
    out_key = f"videos/{batch_id}/annotated.webm"
    out_path = storage.resolve(out_key)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    events, idx, done = [], 0, 0
    last = None
    opts = {**options, "skip_explain": True}
    while not stop.is_set():
        ok, frame = cap.read()
        if not ok:
            break
        if idx in targets:
            ok2, buf = cv2.imencode(".png", frame)
            ts = idx / fps
            with session_scope() as db:
                user = db.get(User, user_id) if user_id else None
                insp = run_inspection(db, data=buf.tobytes(), filename=f"frame_{idx:06d}.png", profile_id=profile_id,
                                      user=user, source="video", batch_id=batch_id, options=opts,
                                      video_timestamp_s=ts)
                last = summary(insp)
            events.append({"frame": idx, "timestamp_s": ts, **{k: last[k] for k in
                                                               ("id", "status", "defect_class", "severity")}})
            done += 1
            MANAGER.update(job_id, progress=done / max(1, len(targets)), message=f"frame {idx} ({done}/{len(targets)})")
            if done >= len(targets):
                pass
        # annotate every frame with the most recent decision (downscaled to <= 720p for size)
        h, w = frame.shape[:2]
        s = min(1.0, 720 / max(h, w))
        fr = cv2.resize(frame, (int(w * s) // 2 * 2, int(h * s) // 2 * 2)) if s < 1 else frame
        if writer is None:
            writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"VP80"), fps, (fr.shape[1], fr.shape[0]))
        if last:
            col = STATUS_BGR.get(last["status"], (200, 200, 200))
            cv2.rectangle(fr, (0, 0), (fr.shape[1], 34), (20, 20, 20), -1)
            label = f"{last['status']}  {last['defect_class'] or ''}  sev={last['severity']}  t={events[-1]['timestamp_s']:.1f}s"
            cv2.putText(fr, label, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
            cv2.rectangle(fr, (0, 0), (fr.shape[1] - 1, fr.shape[0] - 1), col, 4)
        writer.write(fr)
        idx += 1
        if n <= 0 and done >= max_frames:
            break
    cap.release()
    if writer:
        writer.release()
    agg = _aggregate([{**e, "defect_present": e["status"] == "FAIL" or e["defect_class"] is not None,
                       "latency_ms": 0} for e in events])
    with session_scope() as db:
        db.get(ProductionBatch, batch_id).summary = {**agg, "events": events, "annotated_video": out_key,
                                                     "fps": fps, "frames": idx}
    return {"batch_id": batch_id, "events": events, "annotated_video": out_key, "frames_total": idx,
            "frames_inspected": done, "fps": fps}


# ------------------------------------------------------------------ production-line simulation
def simulation_job(job_id: str, stop: threading.Event, profile_id: str, batch_id: str, user_id: int | None,
                   interval_s: float, count: int, defect_rate: float | None, line_id: str):
    """Feeds REAL held-out test images of the profile's dataset through the REAL pipeline at a configured
    arrival rate. Every record is flagged is_simulated=True with the dataset label as ground truth."""
    with session_scope() as db:
        from app.models.entities import ProductProfile
        cfg = db.get(ProductProfile, profile_id).config
        if not db.get(ProductionLine, line_id):
            db.add(ProductionLine(id=line_id, name=f"Simulated line {line_id}", is_simulated=True))
    rows = [r for r in load_manifest(cfg["dataset"])["samples"] if r["split"] == "test"]
    good = [r for r in rows if r["label"] in ("good", "normal")]
    bad = [r for r in rows if r["label"] not in ("good", "normal")]
    rng = random.Random(int(time.time()))
    stats = {"accepted": 0, "rejected": 0, "review": 0, "inconclusive": 0, "latencies": []}
    t_start = time.time()
    for i in range(count):
        if stop.is_set():
            break
        t0 = time.time()
        pool = bad if (good and defect_rate is not None and rng.random() < defect_rate) or not good else good
        if defect_rate is None:
            pool = rows
        r = rng.choice(pool)
        data = absp(r["path"]).read_bytes()
        serial = f"SIM-{batch_id[:6]}-{i + 1:05d}"
        with session_scope() as db:
            user = db.get(User, user_id) if user_id else None
            insp = run_inspection(db, data=data, filename=Path(r["path"]).name, profile_id=profile_id, user=user,
                                  source="simulation", batch_id=batch_id, line_id=line_id, product_serial=serial,
                                  ground_truth=r["label"], is_simulated=True, options={"skip_explain": True})
            s = summary(insp)
        key = {"PASS": "accepted", "FAIL": "rejected", "REVIEW REQUIRED": "review"}.get(s["status"], "inconclusive")
        stats[key] += 1
        stats["latencies"].append(s["latency_ms"])
        elapsed = time.time() - t_start
        ev = {"job_id": job_id, "seq": i + 1, "serial": serial, "status": s["status"], "inspection_id": s["id"],
              "defect_class": s["defect_class"], "ground_truth": r["label"], "latency_ms": s["latency_ms"],
              "throughput_per_min": (i + 1) / elapsed * 60, "signal": "REJECT" if s["status"] == "FAIL" else
              ("HOLD" if s["status"] != "PASS" else "PASS"), "simulated": True,
              **{k: v for k, v in stats.items() if k != "latencies"}}
        bus.publish("simulation", ev)
        MANAGER.update(job_id, progress=(i + 1) / count, message=f"{i + 1}/{count} items",
                       result={k: v for k, v in ev.items() if k != "job_id"})
        time.sleep(max(0.0, interval_s - (time.time() - t0)))
    lat = stats.pop("latencies")
    res = {**stats, "items": sum(stats.values()), "avg_latency_ms": float(np.mean(lat)) if lat else None,
           "throughput_per_min": sum(stats.values()) / max(1e-6, time.time() - t_start) * 60, "simulated": True}
    with session_scope() as db:
        db.get(ProductionBatch, batch_id).summary = res
    return res


# ------------------------------------------------------------------ training (subprocess)
def start_training(tj_id: str, kind: str, config: dict):
    log_dir = ROOT / "storage" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    with session_scope() as db:
        tj = db.get(TrainingJob, tj_id)
        log_path = log_dir / f"training_{tj.run_id}.log"
        if kind == "classification":
            cmd = [sys.executable, str(ROOT / "scripts" / "train_classifiers.py"), "--archs", config["arch"],
                   "--dataset", config.get("dataset", "neu_cls"), "--epochs", str(config.get("epochs", 8)),
                   "--batch-size", str(config.get("batch_size", 32)), "--max-minutes", str(config.get("max_minutes", 30)),
                   "--loss", config.get("loss", "ce"), "--run-id", tj.run_id]
            if config.get("lr"):
                cmd += ["--lr", str(config["lr"])]
            if config.get("resume"):
                cmd += ["--resume"]
        elif kind == "anomaly_custom":
            cmd = [sys.executable, str(ROOT / "scripts" / "train_custom_anomaly.py"), config["slug"],
                   "--run-id", tj.run_id]
        else:
            cmd = [sys.executable, str(ROOT / "scripts" / "train_anomaly.py"), config["category"],
                   "--coreset", str(config.get("coreset", 0.01))]
        env = {**os.environ, "PYTHONIOENCODING": "utf-8", "MLFLOW_DISABLE_AGENT_HINT": "1"}
        f = open(log_path, "w", encoding="utf-8")
        p = subprocess.Popen(cmd, cwd=str(ROOT), stdout=f, stderr=subprocess.STDOUT, env=env)
        MANAGER.procs[tj_id] = p
        tj.pid, tj.status, tj.log_path = p.pid, "running", str(log_path)

    def wait():
        rc = p.wait()
        f.close()
        with session_scope() as db:
            tj = db.get(TrainingJob, tj_id)
            st = json.loads((CHECKPOINT_DIR / tj.run_id / "status.json").read_text()) \
                if (CHECKPOINT_DIR / tj.run_id / "status.json").exists() else {}
            tj.status = st.get("state") if st.get("state") in ("completed", "stopped", "failed") else \
                ("completed" if rc == 0 else "failed")
            tj.message = st.get("message", f"exit code {rc}")
            tj.finished_at = utcnow()
            from app.services import model_registry
            model_registry.sync(db)  # registers as 'validated' only - never deployed automatically
        bus.publish("training", {"id": tj_id, "status": tj.status})
    threading.Thread(target=wait, daemon=True).start()


def stop_training(run_id: str):
    d = CHECKPOINT_DIR / run_id
    d.mkdir(parents=True, exist_ok=True)
    (d / "STOP").write_text("stop requested")
