import { useEffect, useRef, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { api, authUrl } from "../api";
import {
  Badge, Card, ErrorBox, Field, InspectionLink, PageHeader, ProfileSelect, ProgressBar, SeverityBadge, StatusBadge,
  Table, num, pct, useAuth, useInterval, useToast,
} from "../components";
import { ResultPanel } from "./InspectionDetail";

const MODES = [
  ["single", "Single image"],
  ["camera", "Camera"],
  ["batch", "Batch / folder"],
  ["video", "Video"],
  ["compare", "Golden reference"],
];

export default function Inspect() {
  const [params] = useSearchParams();
  const [mode, setMode] = useState(params.get("mode") || "single");
  const [profile, setProfile] = useState(params.get("profile") || localStorage.getItem("qg_profile") || "");
  useEffect(() => {
    if (profile) localStorage.setItem("qg_profile", profile);
  }, [profile]);
  return (
    <>
      <PageHeader title="Inspect — advanced modes" subtitle="Single image with options, live camera, batch/folder, video and golden-reference comparison.">
        <ProfileSelect value={profile} onChange={setProfile} />
        <Link className="btn btn-ghost btn-sm" to="/products/new" title="Train QualityGuard on a new kind of item">Item not listed?</Link>
      </PageHeader>
      <div className="tabs">
        {MODES.map(([k, l]) => (
          <button key={k} className={`tab ${mode === k ? "active" : ""}`} onClick={() => setMode(k)}>{l}</button>
        ))}
      </div>
      {!profile && mode !== "compare" ? (
        <Card><div className="empty">Choose what kind of item you are inspecting (top right) to start.
          <br /><span className="small">Your item isn't in the list? <Link to="/products/new">Add it as a product</Link> using photos of good examples.</span></div></Card>
      ) : (
        <>
          {mode === "single" && <Single profile={profile} />}
          {mode === "camera" && <Camera profile={profile} />}
          {mode === "batch" && <Batch profile={profile} />}
          {mode === "video" && <Video profile={profile} />}
          {mode === "compare" && <Compare />}
        </>
      )}
    </>
  );
}

async function inspectFile(file, profile, extra = {}) {
  const fd = new FormData();
  fd.append("file", file);
  fd.append("profile_id", profile);
  Object.entries(extra).forEach(([k, v]) => v !== undefined && v !== "" && fd.append(k, v));
  return api("/api/inspections", { method: "POST", form: fd });
}

function Single({ profile }) {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState("");
  const [serial, setSerial] = useState("");
  const [explain, setExplain] = useState(true);
  const [cams, setCams] = useState({ gradcam: true, gradcam_pp: false, scorecam: false });
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);
  const [drag, setDrag] = useState(false);

  const pick = (f) => {
    if (!f) return;
    setFile(f);
    setResult(null);
    setPreview(URL.createObjectURL(f));
  };

  const run = async () => {
    setBusy(true);
    setError(null);
    try {
      const options = explain
        ? { cam_methods: Object.keys(cams).filter((k) => cams[k]) }
        : { skip_explain: true };
      if (explain && !options.cam_methods.length) delete options.cam_methods;
      setResult(await inspectFile(file, profile, { product_serial: serial, options: JSON.stringify(options) }));
    } catch (e) {
      setError(e);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="grid grid-3">
      <Card title="Input">
        <div className={`dropzone ${drag ? "drag" : ""}`}
             onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
             onDragLeave={() => setDrag(false)}
             onDrop={(e) => { e.preventDefault(); setDrag(false); pick(e.dataTransfer.files[0]); }}
             onClick={() => document.getElementById("single-file").click()}>
          {preview ? <img src={preview} alt="selected" /> : <span>Drop an image here or click to browse<br /><small className="muted">PNG, JPG, BMP, TIFF, WebP</small></span>}
        </div>
        <input id="single-file" type="file" accept="image/*" hidden onChange={(e) => pick(e.target.files[0])} />
        <Field label="Product serial (optional)">
          <input value={serial} maxLength={128} onChange={(e) => setSerial(e.target.value)} />
        </Field>
        <label className="check"><input type="checkbox" checked={explain} onChange={(e) => setExplain(e.target.checked)} /> Generate explanations</label>
        {explain && (
          <div className="row gap-s wrap small">
            {Object.keys(cams).map((k) => (
              <label key={k} className="check">
                <input type="checkbox" checked={cams[k]} onChange={(e) => setCams({ ...cams, [k]: e.target.checked })} />
                {{ gradcam: "Grad-CAM", gradcam_pp: "Grad-CAM++", scorecam: "Score-CAM (slow)" }[k]}
              </label>
            ))}
          </div>
        )}
        <button className="btn btn-primary btn-block" disabled={!file || busy} onClick={run}>
          {busy ? "Inspecting…" : "Run inspection"}
        </button>
        <ErrorBox error={error} />
      </Card>
      <div className="span-2">
        {result ? <ResultPanel d={result} compact /> : <Card><div className="empty">The result, explanation and severity assessment will appear here.</div></Card>}
      </div>
    </div>
  );
}

function Camera({ profile }) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [stream, setStream] = useState(null);
  const [auto, setAuto] = useState(false);
  const [interval, setIntervalMs] = useState(2000);
  const [busy, setBusy] = useState(false);
  const [last, setLast] = useState(null);
  const [history, setHistory] = useState([]);
  const [error, setError] = useState(null);
  const [devices, setDevices] = useState([]);
  const [deviceId, setDeviceId] = useState(localStorage.getItem("qg_camera") || "");
  const [session, setSession] = useState({ started: null, n: 0, PASS: 0, FAIL: 0, other: 0, latency: 0 });
  const toast = useToast();

  const listDevices = async () => {
    try {
      const all = await navigator.mediaDevices.enumerateDevices();
      setDevices(all.filter((d) => d.kind === "videoinput"));
    } catch {
      setDevices([]);
    }
  };
  useEffect(() => { if (navigator.mediaDevices) listDevices(); }, []);

  const start = async () => {
    setError(null);
    try {
      const video = { width: 1280, height: 720, ...(deviceId ? { deviceId: { exact: deviceId } } : {}) };
      const s = await navigator.mediaDevices.getUserMedia({ video });
      videoRef.current.srcObject = s;
      setStream(s);
      setSession({ started: Date.now(), n: 0, PASS: 0, FAIL: 0, other: 0, latency: 0 });
      listDevices(); // device labels become readable once permission is granted
    } catch (e) {
      setError(new Error(`Camera unavailable: ${e.message}`));
    }
  };
  const stop = () => {
    stream?.getTracks().forEach((t) => t.stop());
    setStream(null);
    setAuto(false);
  };
  useEffect(() => () => stream?.getTracks().forEach((t) => t.stop()), [stream]);

  const capture = async () => {
    if (!stream || busy) return;
    const v = videoRef.current;
    const c = canvasRef.current;
    c.width = v.videoWidth;
    c.height = v.videoHeight;
    c.getContext("2d").drawImage(v, 0, 0);
    const blob = await new Promise((r) => c.toBlob(r, "image/jpeg", 0.92));
    setBusy(true);
    try {
      const r = await inspectFile(new File([blob], `camera_${Date.now()}.jpg`, { type: "image/jpeg" }), profile,
        { source: "camera", options: JSON.stringify({ skip_explain: auto }) });
      setLast(r);
      setHistory((h) => [r, ...h].slice(0, 20));
      setSession((x) => ({ ...x, n: x.n + 1, latency: x.latency + (r.latency_ms || 0),
        PASS: x.PASS + (r.status === "PASS"), FAIL: x.FAIL + (r.status === "FAIL"),
        other: x.other + (r.status !== "PASS" && r.status !== "FAIL") }));
      if (r.status === "FAIL") toast(`Defect: ${r.defect_class || "anomaly"} (${r.severity})`, r.severity === "critical" ? "error" : "warn");
    } catch (e) {
      setError(e);
      setAuto(false);
    } finally {
      setBusy(false);
    }
  };
  useInterval(capture, auto && stream ? interval : null);

  return (
    <div className="grid grid-3">
      <Card title="Live camera" className="span-2" actions={
        <div className="row gap-s">
          <select value={deviceId} disabled={!!stream} aria-label="Camera"
                  onChange={(e) => { setDeviceId(e.target.value); localStorage.setItem("qg_camera", e.target.value); }}>
            <option value="">Default camera</option>
            {devices.map((d, i) => <option key={d.deviceId || i} value={d.deviceId}>{d.label || `Camera ${i + 1}`}</option>)}
          </select>
          {stream ? <button className="btn btn-ghost" onClick={stop}>Stop camera</button> : <button className="btn btn-primary" onClick={start}>Start camera</button>}
        </div>
      }>
        <div className="camera-wrap">
          <video ref={videoRef} autoPlay playsInline muted />
          {last && <div className={`camera-overlay st-${last.status.replace(" ", "-")}`}>{last.status} {last.defect_class || ""}</div>}
        </div>
        <canvas ref={canvasRef} hidden />
        <div className="row gap wrap">
          <button className="btn btn-primary" disabled={!stream || busy} onClick={capture}>{busy ? "Inspecting…" : "Capture & inspect"}</button>
          <label className="check"><input type="checkbox" checked={auto} disabled={!stream} onChange={(e) => setAuto(e.target.checked)} /> Continuous</label>
          <select value={interval} onChange={(e) => setIntervalMs(+e.target.value)}>
            {[1000, 2000, 5000, 10000].map((m) => <option key={m} value={m}>every {m / 1000}s</option>)}
          </select>
        </div>
        <ErrorBox error={error} />
        {session.n > 0 && (
          <div className="kv kv-4">
            <div><span className="muted">Inspected</span><b>{session.n}</b></div>
            <div><span className="muted">PASS / FAIL / other</span><b>{session.PASS} / {session.FAIL} / {session.other}</b></div>
            <div><span className="muted">Throughput</span><b>{num(session.n / Math.max(1, (Date.now() - session.started) / 60000), 1)} / min</b></div>
            <div><span className="muted">Mean latency</span><b>{num(session.latency / session.n, 0)} ms</b></div>
          </div>
        )}
      </Card>
      <Card title="Captured">
        <Table rows={history} empty="No captures yet" columns={[
          { label: "ID", render: (r) => <InspectionLink id={r.id} /> },
          { label: "Status", render: (r) => <StatusBadge status={r.status} /> },
          { label: "Defect", render: (r) => r.defect_class || "—" },
        ]} />
      </Card>
    </div>
  );
}

function useJob(jobId, onDone) {
  const [job, setJob] = useState(null);
  useInterval(async () => {
    if (!jobId || (job && ["completed", "failed", "stopped"].includes(job.status))) return;
    const j = await api(`/api/jobs/${jobId}`);
    setJob(j);
    if (["completed", "failed", "stopped"].includes(j.status)) onDone?.(j);
  }, jobId ? 1200 : null);
  useEffect(() => setJob(null), [jobId]);
  return job;
}

function JobStatus({ job, onStop }) {
  if (!job) return <div className="muted">Queued…</div>;
  return (
    <div className="stack-s">
      <div className="row between"><span><Badge kind={job.status === "failed" ? "fail" : job.status === "completed" ? "pass" : "info"}>{job.status}</Badge> {job.message}</span>
        {["queued", "running"].includes(job.status) && <button className="btn btn-ghost btn-sm" onClick={onStop}>Stop</button>}</div>
      <ProgressBar value={job.progress} />
    </div>
  );
}

function Batch({ profile }) {
  const [files, setFiles] = useState([]);
  const [name, setName] = useState("");
  const [submitted, setSubmitted] = useState(null);
  const [batch, setBatch] = useState(null);
  const [error, setError] = useState(null);
  const toast = useToast();
  const job = useJob(submitted?.job_id, async (j) => {
    setBatch(await api(`/api/batches/${submitted.batch_id}`));
    toast(`Batch ${j.status}`, j.status === "completed" ? "success" : "warn");
  });

  const submit = async (source) => {
    setError(null);
    setBatch(null);
    const fd = new FormData();
    files.forEach((f) => fd.append("files", f, f.webkitRelativePath ? f.webkitRelativePath.replaceAll("/", "_") : f.name));
    fd.append("profile_id", profile);
    fd.append("name", name);
    fd.append("source", source);
    fd.append("options", JSON.stringify({ skip_explain: true }));
    try {
      setSubmitted(await api("/api/inspections/batch", { method: "POST", form: fd }));
    } catch (e) {
      setError(e);
    }
  };
  const isFolder = files.some((f) => f.webkitRelativePath);

  return (
    <div className="grid grid-3">
      <Card title="Batch upload">
        <Field label="Batch name"><input value={name} onChange={(e) => setName(e.target.value)} placeholder="e.g. Lot 2026-11-05 A" /></Field>
        <Field label="Images"><input type="file" multiple accept="image/*" onChange={(e) => setFiles([...e.target.files])} /></Field>
        <Field label="…or a whole folder"><input type="file" webkitdirectory="" directory="" onChange={(e) => setFiles([...e.target.files].filter((f) => f.type.startsWith("image/")))} /></Field>
        <p className="muted small">{files.length} image(s) selected. Explanations are skipped in batch mode for speed; open any item to view details.</p>
        <button className="btn btn-primary btn-block" disabled={!files.length || (job && job.status === "running")} onClick={() => submit(isFolder ? "folder" : "batch")}>Start batch</button>
        <ErrorBox error={error} />
        {submitted && <JobStatus job={job} onStop={() => api(`/api/jobs/${submitted.job_id}/stop`, { method: "POST" })} />}
      </Card>
      <Card title="Batch result" className="span-2" actions={batch && <Link to={`/inspections?batch_id=${batch.id}`}>Open in inspections</Link>}>
        {batch ? (
          <>
            <div className="stats">
              <StatSmall label="Processed" value={batch.summary.total} />
              <StatSmall label="Acceptance" value={pct(batch.summary.acceptance_rate)} />
              <StatSmall label="Avg latency" value={`${num(batch.summary.avg_latency_ms, 0)} ms`} />
              <StatSmall label="Errors" value={(batch.summary.errors || []).length} />
            </div>
            <Table rows={batch.items} columns={[
              { label: "ID", render: (r) => <InspectionLink id={r.id} /> },
              { label: "File", key: "filename" },
              { label: "Status", render: (r) => <StatusBadge status={r.status} /> },
              { label: "Defect", render: (r) => r.defect_class || (r.defect_present ? "anomaly" : "—") },
              { label: "Severity", render: (r) => <SeverityBadge level={r.severity} /> },
              { label: "Quality", render: (r) => num(r.quality_index, 0) },
            ]} />
          </>
        ) : <div className="empty">Results appear when the batch finishes.</div>}
      </Card>
    </div>
  );
}

const StatSmall = ({ label, value }) => (
  <div className="stat"><div className="stat-label">{label}</div><div className="stat-value">{value ?? "—"}</div></div>
);

function Video({ profile }) {
  const [file, setFile] = useState(null);
  const [every, setEvery] = useState(1);
  const [maxFrames, setMaxFrames] = useState(60);
  const [submitted, setSubmitted] = useState(null);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const job = useJob(submitted?.job_id, (j) => setResult(j.result));

  const submit = async () => {
    setError(null);
    setResult(null);
    const fd = new FormData();
    fd.append("file", file);
    fd.append("profile_id", profile);
    fd.append("every_seconds", every);
    fd.append("max_frames", maxFrames);
    try {
      setSubmitted(await api("/api/inspections/video", { method: "POST", form: fd }));
    } catch (e) {
      setError(e);
    }
  };
  return (
    <div className="grid grid-3">
      <Card title="Video inspection">
        <Field label="Video file"><input type="file" accept="video/*" onChange={(e) => setFile(e.target.files[0])} /></Field>
        <div className="row gap">
          <Field label="Sample every (s)"><input type="number" min="0.05" max="60" step="0.05" value={every} onChange={(e) => setEvery(e.target.value)} /></Field>
          <Field label="Max frames"><input type="number" min="1" max="600" value={maxFrames} onChange={(e) => setMaxFrames(e.target.value)} /></Field>
        </div>
        <button className="btn btn-primary btn-block" disabled={!file} onClick={submit}>Analyse video</button>
        <ErrorBox error={error} />
        {submitted && <JobStatus job={job} onStop={() => api(`/api/jobs/${submitted.job_id}/stop`, { method: "POST" })} />}
      </Card>
      <Card title="Timeline" className="span-2">
        {result ? (
          <>
            <video controls className="video" src={authUrl(`/api/batches/${result.batch_id}/video`)} />
            <Table rows={result.events} rowKey={(r) => r.frame} columns={[
              { label: "t (s)", render: (r) => num(r.timestamp_s, 1) },
              { label: "Frame", key: "frame" },
              { label: "ID", render: (r) => <InspectionLink id={r.id} /> },
              { label: "Status", render: (r) => <StatusBadge status={r.status} /> },
              { label: "Defect", render: (r) => r.defect_class || "—" },
              { label: "Severity", render: (r) => <SeverityBadge level={r.severity} /> },
            ]} />
          </>
        ) : <div className="empty">Sampled frames are inspected and an annotated video is produced.</div>}
      </Card>
    </div>
  );
}

function Compare() {
  const [ref, setRef] = useState(null);
  const [test, setTest] = useState(null);
  const [res, setRes] = useState(null);
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);
  const { user } = useAuth();
  const run = async () => {
    setBusy(true);
    setError(null);
    const fd = new FormData();
    fd.append("reference", ref);
    fd.append("test", test);
    try {
      setRes(await api("/api/inspections/compare-reference", { method: "POST", form: fd }));
    } catch (e) {
      setError(e);
    } finally {
      setBusy(false);
    }
  };
  return (
    <div className="grid grid-3">
      <Card title="Golden-sample comparison">
        <p className="muted small">Aligns a test image to a known-good reference and highlights differences. Classical CV — flags any change, including benign ones.</p>
        <Field label="Reference (good) image"><input type="file" accept="image/*" onChange={(e) => setRef(e.target.files[0])} /></Field>
        <Field label="Test image"><input type="file" accept="image/*" onChange={(e) => setTest(e.target.files[0])} /></Field>
        <button className="btn btn-primary btn-block" disabled={!ref || !test || busy || !user} onClick={run}>{busy ? "Comparing…" : "Compare"}</button>
        <ErrorBox error={error} />
      </Card>
      <Card title="Differences" className="span-2">
        {res ? (
          <>
            <div className="kv">
              {Object.entries(res).filter(([k, v]) => k !== "images" && typeof v !== "object").map(([k, v]) => (
                <div key={k}><span className="muted">{k.replaceAll("_", " ")}</span><b>{typeof v === "number" ? num(v, 3) : String(v)}</b></div>
              ))}
            </div>
            <div className="gallery">
              {Object.entries(res.images).map(([k, src]) => (
                <figure key={k}><img src={src} alt={k} /><figcaption>{k.replaceAll("_", " ")}</figcaption></figure>
              ))}
            </div>
          </>
        ) : <div className="empty">Upload a reference and a test image.</div>}
      </Card>
    </div>
  );
}
