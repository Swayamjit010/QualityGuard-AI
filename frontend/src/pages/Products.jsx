import { useState } from "react";
import { Link, Route, Routes, useNavigate, useParams } from "react-router-dom";
import { api, authUrl } from "../api";
import {
  Async, Badge, Card, Empty, ErrorBox, Field, PageHeader, ProgressBar, can, num, pct, useApi, useAuth, useInterval,
  useToast,
} from "../components";

export default function Products() {
  return (
    <Routes>
      <Route index element={<ProductList />} />
      <Route path="new" element={<NewProduct />} />
      <Route path=":slug" element={<ProductDetail />} />
    </Routes>
  );
}

const STATE = {
  ready: ["pass", "Ready to inspect"], trained: ["review", "Trained — waiting for approval"],
  running: ["info", "Training…"], failed: ["fail", "Training failed"], "not started": ["muted", "Not trained"],
};

function ProductList() {
  const { user } = useAuth();
  const state = useApi("/api/products");
  useInterval(() => state.data?.some((p) => p.state === "running") && state.reload(), 4000);
  return (
    <>
      <PageHeader title="Products" subtitle="The kinds of items QualityGuard can inspect. Add your own from photos of good items.">
        {can(user, "train") && <Link className="btn btn-primary" to="/products/new">+ Add a product</Link>}
      </PageHeader>
      <Async state={state}>
        {(rows) => (
          <div className="product-grid">
            {rows.map((p) => {
              const [kind, label] = STATE[p.state] || ["muted", p.state];
              return (
                <Card key={p.id} className="product-card">
                  {p.thumbnails?.length > 0 && <img className="product-thumb" src={authUrl(p.thumbnails[0])} alt={`${p.name} example`} />}
                  <div className="row between"><h3>{p.name}</h3><Badge kind={kind}>{label}</Badge></div>
                  <p className="muted small">{p.description || (p.kind === "built-in" ? "Built-in product" : "")}</p>
                  {p.kind === "built-in" && (
                    <p className="small">Engine: {p.engines.join(" + ")}{p.defect_classes?.length ? ` · knows: ${p.defect_classes.join(", ")}` : ""}</p>
                  )}
                  {p.kind === "custom" && <p className="small">{p.counts.good} good photo(s){p.counts.defect ? `, ${p.counts.defect} defective` : ""}</p>}
                  {p.state === "running" && <ProgressBar value={p.progress || 0} />}
                  <div className="row gap-s">
                    {(p.state === "ready" || p.inspectable) && <Link className="btn btn-primary btn-sm" to={`/inspect?profile=${p.profile_id}`}>Inspect an item</Link>}
                    {p.kind === "custom" && <Link className="btn btn-ghost btn-sm" to={`/products/${p.slug}`}>Details</Link>}
                    {p.kind === "built-in" && <Link className="btn btn-ghost btn-sm" to="/profiles">Rules</Link>}
                  </div>
                </Card>
              );
            })}
          </div>
        )}
      </Async>
    </>
  );
}

function Drop({ label, hint, files, setFiles, id }) {
  const [drag, setDrag] = useState(false);
  const add = (list) => setFiles((f) => [...f, ...[...list].filter((x) => x.type.startsWith("image/"))]);
  return (
    <Field label={label} hint={hint}>
      <div className={`dropzone dropzone-sm ${drag ? "drag" : ""}`}
           onDragOver={(e) => { e.preventDefault(); setDrag(true); }} onDragLeave={() => setDrag(false)}
           onDrop={(e) => { e.preventDefault(); setDrag(false); add(e.dataTransfer.files); }}
           onClick={() => document.getElementById(id).click()}>
        <span>{files.length ? `${files.length} photo(s) selected — click or drop to add more` : "Drop photos here or click to choose"}</span>
      </div>
      <div className="row gap-s">
        <input id={id} type="file" accept="image/*" multiple hidden onChange={(e) => add(e.target.files)} />
        <label className="btn btn-ghost btn-sm">Choose a folder
          <input type="file" webkitdirectory="" directory="" hidden onChange={(e) => add(e.target.files)} />
        </label>
        {files.length > 0 && <button type="button" className="btn btn-ghost btn-sm" onClick={() => setFiles([])}>Clear</button>}
      </div>
      {files.length > 0 && (
        <div className="thumbs">{files.slice(0, 12).map((f, i) => <img key={i} src={URL.createObjectURL(f)} alt="" />)}
          {files.length > 12 && <span className="muted small">+{files.length - 12} more</span>}</div>
      )}
    </Field>
  );
}

function NewProduct() {
  const [name, setName] = useState("");
  const [desc, setDesc] = useState("");
  const [good, setGood] = useState([]);
  const [defect, setDefect] = useState([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const navigate = useNavigate();
  const submit = async (e) => {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      const fd = new FormData();
      fd.append("name", name);
      fd.append("description", desc);
      good.forEach((f) => fd.append("good", f));
      defect.forEach((f) => fd.append("defect", f));
      const r = await api("/api/products", { method: "POST", form: fd });
      navigate(`/products/${r.slug}`);
    } catch (err) {
      setError(err);
    } finally {
      setBusy(false);
    }
  };
  return (
    <>
      <PageHeader title="Add a product" subtitle="Teach QualityGuard a new kind of item using photos of good examples. No defect labels needed.">
        <Link className="btn btn-ghost" to="/products">← All products</Link>
      </PageHeader>
      <div className="grid grid-3">
        <Card title="1 · Describe the item">
          <form id="new-product" onSubmit={submit} className="stack-s">
            <Field label="Product name"><input value={name} onChange={(e) => setName(e.target.value)} placeholder="e.g. Brass hex nut, PCB board, ceramic tile" required maxLength={64} /></Field>
            <Field label="Description (optional)"><textarea rows={3} value={desc} onChange={(e) => setDesc(e.target.value)} maxLength={500} /></Field>
          </form>
          <h4>Tips for good results</h4>
          <ul className="small">
            <li>Use the <b>same camera position, distance, background and lighting</b> you will use when inspecting.</li>
            <li>One item per photo, roughly centred and filling the frame.</li>
            <li>At least 10 photos; <b>30–100 is much better</b>. Include the normal variation of good items.</li>
            <li>Defective examples are optional — they are only used to tell you how many defects the model catches.</li>
          </ul>
        </Card>
        <Card title="2 · Upload photos" className="span-2">
          <Drop id="good-files" label="Photos of GOOD items (required)" hint="10 minimum, 30+ recommended" files={good} setFiles={setGood} />
          <Drop id="defect-files" label="Photos of DEFECTIVE items (optional)" hint="used only to measure detection — never to tune the model" files={defect} setFiles={setDefect} />
          <ErrorBox error={error} />
          <button className="btn btn-primary" form="new-product" disabled={busy || !name.trim() || good.length < 10}>
            {busy ? "Uploading…" : good.length < 10 ? `Add ${10 - good.length} more good photo(s)` : `Create and train (${good.length} good${defect.length ? `, ${defect.length} defective` : ""})`}
          </button>
          <p className="muted small">Training runs in the background, typically 1–5 minutes on a CPU. You can leave this page.</p>
        </Card>
      </div>
    </>
  );
}

function ProductDetail() {
  const { slug } = useParams();
  const { user } = useAuth();
  const toast = useToast();
  const state = useApi(`/api/products/${slug}`);
  const [busy, setBusy] = useState(false);
  const [more, setMore] = useState([]);
  const [moreBad, setMoreBad] = useState([]);
  useInterval(() => ["running", "not started"].includes(state.data?.state) && state.reload(), 3000);
  const activate = async () => {
    setBusy(true);
    try {
      await api(`/api/products/${slug}/activate`, { method: "POST" });
      toast("Product activated — you can inspect it now", "success");
      state.reload();
    } catch (e) {
      toast(e.message, "error");
    } finally {
      setBusy(false);
    }
  };
  const retrain = async () => {
    setBusy(true);
    try {
      const fd = new FormData();
      more.forEach((f) => fd.append("good", f));
      moreBad.forEach((f) => fd.append("defect", f));
      await api(`/api/products/${slug}/photos`, { method: "POST", form: fd });
      setMore([]);
      setMoreBad([]);
      toast("Retraining started", "success");
      state.reload();
    } catch (e) {
      toast(e.message, "error");
    } finally {
      setBusy(false);
    }
  };
  return (
    <Async state={state}>
      {(p) => {
        const [kind, label] = STATE[p.state] || ["muted", p.state];
        const s = p.summary;
        return (
          <>
            <PageHeader title={p.name} subtitle={p.description || "Custom product"}>
              <Badge kind={kind}>{label}</Badge>
              <Link className="btn btn-ghost" to="/products">← All products</Link>
              {(p.state === "ready" || p.inspectable) && <Link className="btn btn-primary" to={`/inspect?profile=${p.profile_id}`}>Inspect an item</Link>}
            </PageHeader>
            <div className="grid grid-3">
              <Card title="Training" className="span-2">
                {(p.state === "running" || p.state === "not started") && (
                  <div className="stack-s"><ProgressBar value={p.progress || 0} /><span className="muted">{p.message || "queued…"}</span>
                    {p.inspectable && <span className="small">The current model ({p.active_run}) keeps working until the new version is approved.</span>}</div>
                )}
                {p.state === "failed" && <div className="error-box">{p.message}</div>}
                {s && (
                  <>
                    <div className="kv kv-4">
                      <div><span className="muted">Good photos</span><b>{s.good_photos}</b></div>
                      <div><span className="muted">Calibration</span><b>{s.calibration_folds}-fold</b></div>
                      <div><span className="muted">Defect photos caught</span><b>{s.defect_photos ? pct(s.defect_detection_rate?.patchcore, 0) : "— (none given)"}</b></div>
                      <div><span className="muted">Good photos wrongly flagged</span><b>{pct(s.good_false_reject_rate?.patchcore, 0)}</b></div>
                      <div><span className="muted">Other products recognised as unfamiliar</span><b>{s.unfamiliar?.foreign_detection_rate != null ? pct(s.unfamiliar.foreign_detection_rate, 0) : "—"}</b></div>
                      <div><span className="muted">Training time</span><b>{s.training_seconds} s</b></div>
                      <div><span className="muted">Scoring time</span><b>{num(s.latency_ms?.patchcore, 0)} ms</b></div>
                      <div><span className="muted">Model</span><b className="small">{p.run_id}</b></div>
                    </div>
                    {s.fusion?.defects_auto_failed != null && (
                      <p className="small">Of the {s.defect_photos} defective photos: {s.fusion.defects_auto_failed} would be rejected automatically,
                        {" "}{s.fusion.defects_sent_to_review} sent to review, {s.fusion.defects_auto_passed} missed.</p>
                    )}
                    {s.warnings?.map((w, i) => <div key={i} className="info-box">{w}</div>)}
                    <p className="muted small">Thresholds come only from the good photos (out-of-fold calibration); defective photos are used only for the figures above.</p>
                  </>
                )}
                {p.state === "trained" && (
                  can(user, "approve_models")
                    ? <button className="btn btn-primary" disabled={busy} onClick={activate}>Approve & activate this model</button>
                    : <div className="info-box">Trained. A quality manager or administrator must approve it before it can be used.</div>
                )}
                {p.inspectable && p.active_run && <p className="small">Active model: <code>{p.active_run}</code> · <Link to={`/anomaly`}>thresholds & score distribution</Link> · <Link to="/models">model registry</Link></p>}
              </Card>
              <Card title="Example good photos">
                {p.thumbnails.length ? <div className="gallery gallery-sm">{p.thumbnails.map((t) => <figure key={t}><img src={authUrl(t)} alt="good example" /></figure>)}</div> : <Empty>No photos</Empty>}
              </Card>
              {can(user, "train") && p.state !== "running" && (
                <Card title="Improve the model" className="span-3">
                  <p className="small">Add more photos (for example of good items that were wrongly flagged) and retrain. The current model keeps working until the new one is approved.</p>
                  <div className="grid grid-2">
                    <Drop id="more-good" label="More good photos" files={more} setFiles={setMore} />
                    <Drop id="more-bad" label="More defective photos (optional)" files={moreBad} setFiles={setMoreBad} />
                  </div>
                  <button className="btn btn-ghost" disabled={busy || (!more.length && !moreBad.length)} onClick={retrain}>Add photos and retrain</button>
                </Card>
              )}
            </div>
          </>
        );
      }}
    </Async>
  );
}
