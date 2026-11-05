import { useState } from "react";
import { Link, Route, Routes, useNavigate, useParams } from "react-router-dom";
import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api, authUrl } from "../api";
import {
  Async, Badge, Card, Empty, ErrorBox, Field, JsonView, Modal, PageHeader, ProfileSelect, ProgressBar, Table, can, dt,
  num, pct, useApi, useAuth, useInterval, useToast,
} from "../components";

const STATUS_KIND = { deployed: "pass", approved: "info", validated: "review", retired: "muted", failed: "fail" };

export default function Models() {
  return (
    <Routes>
      <Route index element={<ModelList />} />
      <Route path=":id" element={<ModelDetail />} />
    </Routes>
  );
}

function ModelList() {
  const state = useApi("/api/models");
  const { user } = useAuth();
  const navigate = useNavigate();
  const toast = useToast();
  const [tab, setTab] = useState("classification");
  const sync = async () => {
    try {
      const r = await api("/api/models/sync", { method: "POST" });
      toast(`Registry synced: ${r.added.length} added, ${r.updated.length} updated`, "success");
      state.reload();
    } catch (e) {
      toast(e.message, "error");
    }
  };
  return (
    <>
      <PageHeader title="Models & training" subtitle="Versioned model registry with approval-gated deployment.">
        {can(user, "manage_models") && <button className="btn btn-ghost" onClick={sync}>Sync registry</button>}
      </PageHeader>
      <div className="tabs">
        <button className={`tab ${tab === "classification" ? "active" : ""}`} onClick={() => setTab("classification")}>Classifiers</button>
        <button className={`tab ${tab === "anomaly" ? "active" : ""}`} onClick={() => setTab("anomaly")}>Anomaly detectors</button>
        <button className={`tab ${tab === "other" ? "active" : ""}`} onClick={() => setTab("other")}>Segmentation & recognition</button>
        <button className={`tab ${tab === "training" ? "active" : ""}`} onClick={() => setTab("training")}>Training</button>
      </div>
      {tab === "training" ? <Training onDone={state.reload} /> : (
        <Card title={tab === "classification" ? "Classifiers (compare only within the same dataset / held-out split)" :
          tab === "anomaly" ? "Anomaly detectors (per product category)" : "Supervised segmentation & component recognition"}>
          <Async state={state}>
            {(rows) => {
              const r = rows.filter((m) => (tab === "classification" ? m.task === "classification" :
                tab === "anomaly" ? m.task === "anomaly_detection" : ["segmentation", "component_recognition"].includes(m.task)));
              if (tab === "other") return (
                <Table rows={r} empty="No segmentation or recognition models registered." onRowClick={(m) => navigate(`/models/${m.id}`)} columns={[
                  { label: "Run", render: (m) => <span className="mono small">{m.run_id}</span> },
                  { label: "Task", key: "task" },
                  { label: "Dataset", render: (m) => m.dataset || "multiple" },
                  { label: "Status", render: (m) => <Badge kind={STATUS_KIND[m.status]}>{m.status}</Badge> },
                  { label: "Image AUROC / top-1", render: (m) => num(m.metrics.image_auroc ?? m.metrics.top1_accuracy, 4) },
                  { label: "Recall / false-reject", render: (m) => m.task === "segmentation" ? num(m.metrics.image_recall, 3) : pct(m.metrics.false_reject_rate) },
                  { label: "Dice / foreign rejected", render: (m) => m.task === "segmentation" ? num(m.metrics.mean_dice_defective, 3) :
                    pct(m.metrics.rejected_rate_imagenette_natural_photos) },
                  { label: "Pixel AUROC", render: (m) => num(m.metrics.pixel_auroc, 4) },
                ]} />
              );
              const best = tab === "classification" ? Math.max(...r.filter((m) => m.dataset === "neu_cls").map((m) => m.metrics.f1_macro ?? 0)) : null;
              return tab === "classification" ? (
                <Table rows={r} empty="No classifiers registered." onRowClick={(m) => navigate(`/models/${m.id}`)} columns={[
                  { label: "Run", render: (m) => <span className="mono small">{m.run_id}</span> },
                  { label: "Architecture", key: "architecture" },
                  { label: "Dataset", key: "dataset" },
                  { label: "Status", render: (m) => <Badge kind={STATUS_KIND[m.status]}>{m.status}</Badge> },
                  { label: "Accuracy", render: (m) => pct(m.metrics.accuracy) },
                  { label: "Macro F1", render: (m) => <b className={m.metrics.f1_macro === best ? "best" : ""}>{num(m.metrics.f1_macro, 4)}</b> },
                  { label: "Precision", render: (m) => num(m.metrics.precision_macro, 4) },
                  { label: "Recall", render: (m) => num(m.metrics.recall_macro, 4) },
                  { label: "Min class recall", render: (m) => num(m.metrics.min_class_recall, 3) },
                  { label: "ROC-AUC", render: (m) => num(m.metrics.roc_auc_macro_ovr, 4) },
                  { label: "ECE", render: (m) => num(m.metrics.ece, 3) },
                  { label: "Latency (ms)", render: (m) => num(m.metrics.latency_ms_median, 1) },
                  { label: "Size (MB)", render: (m) => num(m.checkpoint_size_mb, 1) },
                ]} />
              ) : (
                <Table rows={r} empty="No anomaly models registered." onRowClick={(m) => navigate(`/models/${m.id}`)} columns={[
                  { label: "Run", render: (m) => <span className="mono small">{m.run_id}</span> },
                  { label: "Dataset", key: "dataset" },
                  { label: "Status", render: (m) => <Badge kind={STATUS_KIND[m.status]}>{m.status}</Badge> },
                  { label: "PatchCore AUROC", render: (m) => num(m.metrics.patchcore_image_auroc, 4) },
                  { label: "PatchCore pixel AUROC", render: (m) => num(m.metrics.patchcore_pixel_auroc, 4) },
                  { label: "PaDiM AUROC", render: (m) => num(m.metrics.padim_image_auroc, 4) },
                  { label: "Auto-decision acc.", render: (m) => pct(m.metrics.fusion_auto_accuracy) },
                  { label: "Review rate", render: (m) => pct(m.metrics.fusion_review_rate) },
                  { label: "Latency (ms)", render: (m) => num(m.metrics.patchcore_latency_ms, 0) },
                ]} />
              );
            }}
          </Async>
          <p className="muted small">Selection should weigh macro-F1 and worst-class recall (missed defects) against latency and size. New versions register as <b>validated</b>; a Quality Manager must approve before deployment.</p>
        </Card>
      )}
    </>
  );
}

function ModelDetail() {
  const { id } = useParams();
  const state = useApi(`/api/models/${id}`);
  const { user } = useAuth();
  const toast = useToast();
  const [deploy, setDeploy] = useState(false);
  const [showEval, setShowEval] = useState(false);
  const act = async (action) => {
    const note = window.prompt(`Note for ${action} (optional)`) ?? null;
    if (note === null) return;
    try {
      await api(`/api/models/${id}/${action}`, { method: "POST", body: { note } });
      toast(`Model ${action}d`, "success");
      state.reload();
    } catch (e) {
      toast(e.message, "error");
    }
  };
  return (
    <Async state={state}>
      {(m) => (
        <>
          <PageHeader title={<span className="mono">{m.run_id}</span>} subtitle={`${m.task} · ${m.architecture} · dataset ${m.dataset} ${m.dataset_version} · created ${dt(m.created_at)}`}>
            <Badge kind={STATUS_KIND[m.status]}>{m.status}</Badge>
            {can(user, "approve_models") && ["validated", "retired"].includes(m.status) && <button className="btn btn-primary" onClick={() => act("approve")}>Approve</button>}
            {can(user, "approve_models") && ["approved", "deployed"].includes(m.status) && <button className="btn btn-primary" onClick={() => setDeploy(true)}>Deploy to profile</button>}
            {can(user, "manage_models") && m.status !== "retired" && <button className="btn btn-ghost" onClick={() => act("retire")}>Retire</button>}
            <Link className="btn btn-ghost" to="/models">← All models</Link>
          </PageHeader>
          <div className="grid grid-3">
            <Card title="Test-set metrics">
              <div className="kv">
                {Object.entries(m.metrics).map(([k, v]) => <div key={k}><span className="muted">{k.replaceAll("_", " ")}</span><b>{num(v, 4)}</b></div>)}
              </div>
              {m.used_by_profiles?.length > 0 && <p className="small">Active in: {m.used_by_profiles.map((p) => <Link key={p} to={`/inspect?profile=${p}`} title="Inspect with this product"><Badge kind="pass">{p}</Badge></Link>)}</p>}
              {m.evaluation && <button className="btn btn-ghost btn-sm" onClick={() => setShowEval(true)}>Full evaluation JSON</button>}
            </Card>
            <Card title="Training history" className="span-2">
              {m.history?.length ? (
                <ResponsiveContainer width="100%" height={260}>
                  <LineChart data={m.history}>
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--grid)" />
                    <XAxis dataKey="epoch" fontSize={11} />
                    <YAxis yAxisId="l" fontSize={11} />
                    <YAxis yAxisId="r" orientation="right" domain={[0, 1]} fontSize={11} />
                    <Tooltip formatter={(v) => num(v, 4)} />
                    <Legend />
                    <Line yAxisId="l" dataKey="train_loss" name="train loss" stroke="#1d6fe0" dot={false} />
                    <Line yAxisId="l" dataKey="val_loss" name="val loss" stroke="#e11d48" dot={false} />
                    <Line yAxisId="r" dataKey="train_acc" name="train acc" stroke="#14b8a6" strokeDasharray="4 3" dot={false} />
                    <Line yAxisId="r" dataKey="val_acc" name="val acc" stroke="#f59e0b" strokeDasharray="4 3" dot={false} />
                  </LineChart>
                </ResponsiveContainer>
              ) : <Empty>No epoch history (anomaly models are fitted, not trained by epochs).</Empty>}
            </Card>
          </div>
          {m.plots?.length > 0 && (
            <Card title="Evaluation plots">
              <div className="gallery gallery-lg">
                {m.plots.map((p) => (
                  <figure key={p}><img src={authUrl(`/api/models/${m.id}/plots/${p}`)} alt={p} /><figcaption>{p.replace(".png", "").replaceAll("_", " ")}</figcaption></figure>
                ))}
              </div>
            </Card>
          )}
          {m.classes?.length > 0 && <Card title="Classes"><div className="row gap-s wrap">{m.classes.map((c) => <Badge key={c}>{c}</Badge>)}</div></Card>}
          {m.notes && <Card title="Notes"><pre className="notes">{m.notes}</pre></Card>}
          {deploy && <DeployModal m={m} onClose={() => setDeploy(false)} onDone={() => { setDeploy(false); state.reload(); }} />}
          {showEval && <Modal title="Evaluation" wide onClose={() => setShowEval(false)}><JsonView value={m.evaluation} /></Modal>}
        </>
      )}
    </Async>
  );
}

function DeployModal({ m, onClose, onDone }) {
  const [profile, setProfile] = useState("");
  const [mode, setMode] = useState(m.task === "classification" ? "primary" : "replace");
  const [error, setError] = useState(null);
  const toast = useToast();
  const go = async () => {
    try {
      const r = await api(`/api/models/${m.id}/activate`, { method: "POST", body: { profile_id: profile, mode } });
      toast(`Deployed to ${r.profile_id} (profile v${r.profile_version})`, "success");
      onDone();
    } catch (e) {
      setError(e);
    }
  };
  return (
    <Modal title="Deploy model version" onClose={onClose}>
      <Field label="Product profile"><ProfileSelect value={profile} onChange={setProfile} /></Field>
      <Field label="Mode" hint="primary: first classifier; ensemble: add to ensemble; replace: replace all models of this task">
        <select value={mode} onChange={(e) => setMode(e.target.value)}>
          <option value="primary">primary</option><option value="ensemble">ensemble</option><option value="replace">replace</option>
        </select>
      </Field>
      <p className="muted small">A new profile revision is recorded, so the change can be rolled back from the Profiles page.</p>
      <ErrorBox error={error} />
      <button className="btn btn-primary" disabled={!profile} onClick={go}>Deploy</button>
    </Modal>
  );
}

function Training({ onDone }) {
  const { user } = useAuth();
  const archs = useApi("/api/models/architectures");
  const jobs = useApi("/api/training/jobs");
  const [f, setF] = useState({ kind: "classification", arch: "mobilenet_v2", dataset: "neu_cls", category: "bottle", epochs: 8, batch_size: 32, lr: "", loss: "ce", max_minutes: 30 });
  const [error, setError] = useState(null);
  const [open, setOpen] = useState(null);
  const toast = useToast();
  const set = (k) => (e) => setF({ ...f, [k]: e.target.value });
  const active = (jobs.data || []).some((j) => ["queued", "running"].includes(j.status));
  useInterval(() => jobs.reload(), active ? 4000 : null);

  const start = async (resume_run_id) => {
    setError(null);
    try {
      const body = { ...f, epochs: +f.epochs, batch_size: +f.batch_size, max_minutes: +f.max_minutes, lr: f.lr ? +f.lr : null };
      if (resume_run_id) Object.assign(body, { resume_run_id, arch: resume_run_id.split("-")[1] });
      const r = await api("/api/training/start", { method: "POST", body });
      toast(`Training ${r.run_id} started`, "success");
      jobs.reload();
    } catch (e) {
      setError(e);
    }
  };
  const stop = async (j) => {
    await api(`/api/training/jobs/${j.id}/stop`, { method: "POST" }).catch((e) => toast(e.message, "error"));
    jobs.reload();
  };

  return (
    <div className="grid grid-3">
      <Card title="Start training">
        {!can(user, "train") ? <Empty>You need the <b>train</b> permission.</Empty> : (
          <>
            <Field label="Kind">
              <select value={f.kind} onChange={set("kind")}>
                <option value="classification">Classifier (supervised, transfer learning)</option>
                <option value="anomaly">Anomaly detector (PatchCore + PaDiM)</option>
              </select>
            </Field>
            {f.kind === "classification" ? (
              <>
                <Field label="Architecture">
                  <select value={f.arch} onChange={set("arch")}>
                    {(archs.data || []).map((a) => <option key={a.name} value={a.name}>{a.name}{a.edge_friendly ? " (edge)" : ""}</option>)}
                  </select>
                </Field>
                <Field label="Dataset"><input value={f.dataset} onChange={set("dataset")} /></Field>
                <div className="row gap">
                  <Field label="Epochs"><input type="number" min="1" max="100" value={f.epochs} onChange={set("epochs")} /></Field>
                  <Field label="Batch"><input type="number" min="4" max="256" value={f.batch_size} onChange={set("batch_size")} /></Field>
                </div>
                <div className="row gap">
                  <Field label="LR (blank = default)"><input value={f.lr} onChange={set("lr")} placeholder="1e-3" /></Field>
                  <Field label="Loss"><select value={f.loss} onChange={set("loss")}><option value="ce">cross-entropy</option><option value="focal">focal</option></select></Field>
                </div>
                <Field label="Time budget (min)"><input type="number" min="1" max="600" value={f.max_minutes} onChange={set("max_minutes")} /></Field>
              </>
            ) : (
              <Field label="MVTec category"><input value={f.category} onChange={set("category")} /></Field>
            )}
            <button className="btn btn-primary btn-block" disabled={active} onClick={() => start()}>{active ? "A job is running" : "Start"}</button>
            <ErrorBox error={error} />
          </>
        )}
      </Card>
      <Card title="Training jobs" className="span-2">
        <Async state={jobs}>
          {(rows) => (
            <Table rows={rows} empty="No training jobs started from the UI yet." onRowClick={(j) => setOpen(j.id)} columns={[
              { label: "Run", render: (j) => <span className="mono small">{j.run_id}</span> },
              { label: "Kind", key: "kind" },
              { label: "Status", render: (j) => <Badge kind={j.status === "completed" ? "pass" : j.status === "failed" ? "fail" : "info"}>{j.status}</Badge> },
              { label: "Progress", render: (j) => j.live?.epochs ? <>{j.live.epoch}/{j.live.epochs} <ProgressBar value={j.live.epoch / j.live.epochs} /></> : j.live?.message || "—" },
              { label: "Val acc", render: (j) => pct(j.live?.val_acc) },
              { label: "Started", render: (j) => dt(j.created_at) },
              { label: "", render: (j) => (
                <div className="row gap-s" onClick={(e) => e.stopPropagation()}>
                  {can(user, "train") && ["queued", "running"].includes(j.status) && <button className="btn btn-ghost btn-sm" onClick={() => stop(j)}>Stop</button>}
                  {can(user, "train") && j.status === "stopped" && j.kind === "classification" && <button className="btn btn-ghost btn-sm" onClick={() => start(j.run_id)}>Resume</button>}
                </div>
              ) },
            ]} />
          )}
        </Async>
      </Card>
      {open && <JobModal id={open} onClose={() => { setOpen(null); onDone(); }} />}
    </div>
  );
}

function JobModal({ id, onClose }) {
  const state = useApi(`/api/training/jobs/${id}`);
  useInterval(() => state.reload(), state.data && ["queued", "running"].includes(state.data.status) ? 3000 : null);
  return (
    <Modal title="Training job" wide onClose={onClose}>
      <Async state={state}>
        {(j) => (
          <>
            <div className="kv kv-4">
              <div><span className="muted">Run</span><b className="mono small">{j.run_id}</b></div>
              <div><span className="muted">Status</span><b>{j.status}</b></div>
              <div><span className="muted">Message</span><b>{j.live?.message || j.message || "—"}</b></div>
              <div><span className="muted">Finished</span><b>{dt(j.finished_at)}</b></div>
            </div>
            {j.history?.length > 1 && (
              <ResponsiveContainer width="100%" height={200}>
                <LineChart data={j.history}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--grid)" />
                  <XAxis dataKey="epoch" fontSize={11} />
                  <YAxis fontSize={11} />
                  <Tooltip formatter={(v) => num(v, 4)} />
                  <Line dataKey="train_loss" stroke="#1d6fe0" dot={false} />
                  <Line dataKey="val_loss" stroke="#e11d48" dot={false} />
                  <Line dataKey="val_acc" stroke="#f59e0b" dot={false} />
                </LineChart>
              </ResponsiveContainer>
            )}
            <h4>Log</h4>
            <pre className="log">{(j.log_tail || []).join("\n") || "(no output yet)"}</pre>
          </>
        )}
      </Async>
    </Modal>
  );
}
