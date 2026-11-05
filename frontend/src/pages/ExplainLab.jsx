import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { Bar, BarChart, CartesianGrid, Legend, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api, authUrl } from "../api";
import {
  Async, Badge, Card, Empty, ErrorBox, Field, PALETTE, PageHeader, ProfileSelect, StatusBadge, Table, can, dt, num, pct,
  short, useApi, useAuth,
} from "../components";

// Every image on this page is a file written by the real inference run of the selected inspection.
const LABELS = {
  original: "Input image", gradcam_overlay: "Grad-CAM", gradcam_heatmap: "Grad-CAM (heat-map only)",
  gradcam_pp_overlay: "Grad-CAM++", gradcam_pp_heatmap: "Grad-CAM++ (heat-map only)", scorecam_overlay: "Score-CAM",
  scorecam_heatmap: "Score-CAM (heat-map only)", anomaly_patchcore_overlay: "PatchCore anomaly map",
  anomaly_patchcore_heatmap: "PatchCore (heat-map only)", anomaly_padim_overlay: "PaDiM anomaly map",
  anomaly_padim_heatmap: "PaDiM (heat-map only)", segmentation_overlay: "Estimated defect region",
  segmentation_mask: "Segmentation mask", bounding_boxes: "Region boxes", defect_zoom: "Largest region (zoom)",
  comparison: "Before / after",
};
const KIND_NOTE = (k) =>
  k.startsWith("gradcam") || k.startsWith("scorecam")
    ? "Explanation map: where the classifier's evidence for the predicted class is concentrated. It is NOT a defect boundary."
    : k.startsWith("anomaly")
      ? "Anomaly map: per-pixel deviation from defect-free reference images."
      : k.startsWith("segmentation") || k === "bounding_boxes" || k === "defect_zoom"
        ? "Thresholded anomaly map (validated pixel threshold): an estimated defect region, not a hand-annotated mask."
        : "";

export default function ExplainLab() {
  const { user } = useAuth();
  const [iid, setIid] = useState(new URLSearchParams(window.location.search).get("inspection") || "");
  const [profile, setProfile] = useState("");
  const recent = useApi("/api/inspections", { query: { page_size: 60, profile_id: profile } }, [profile]);
  const [detail, setDetail] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!iid) return setDetail(null);
    setError(null);
    api(`/api/inspections/${iid}`).then(setDetail).catch(setError);
  }, [iid]);

  return (
    <>
      <PageHeader title="Explainable AI lab"
                  subtitle="Inspect why a model decided what it did. Every map shown was produced by the actual inference run.">
        <ProfileSelect value={profile} onChange={setProfile} allowAll />
      </PageHeader>
      <div className="grid grid-3">
        <Card title="Choose an inspection">
          <Async state={recent}>
            {(r) => r.items.length === 0 ? <Empty>No inspections yet.</Empty> : (
              <select value={iid} onChange={(e) => setIid(e.target.value)} aria-label="Inspection">
                <option value="">— select —</option>
                {r.items.map((i) => (
                  <option key={i.id} value={i.id}>
                    {short(i.id)} · {i.profile_id} · {i.status} · {i.defect_class || (i.defect_present ? "anomaly" : "none")}
                  </option>
                ))}
              </select>
            )}
          </Async>
          <p className="muted small">Older records: open any inspection and use “Open in XAI lab”, or paste its id.</p>
          <input placeholder="Inspection id" value={iid} onChange={(e) => setIid(e.target.value.trim())} aria-label="Inspection id" />
        </Card>
        {can(user, "inspect") && <RunNew onDone={(id) => setIid(id)} />}
      </div>
      <ErrorBox error={error} />
      {detail && <Analysis d={detail} />}
    </>
  );
}

function RunNew({ onDone }) {
  const [file, setFile] = useState(null);
  const [profile, setProfile] = useState("");
  const [methods, setMethods] = useState({ gradcam: true, gradcam_pp: true, scorecam: true });
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const run = async () => {
    setBusy(true);
    setError(null);
    try {
      const fd = new FormData();
      fd.append("file", file);
      fd.append("profile_id", profile);
      const cams = Object.keys(methods).filter((k) => methods[k]);
      fd.append("options", JSON.stringify(cams.length ? { cam_methods: cams } : {}));
      const r = await api("/api/inspections", { method: "POST", form: fd });
      onDone(r.id);
    } catch (e) {
      setError(e);
    } finally {
      setBusy(false);
    }
  };
  return (
    <Card title="Run a new inspection with all explanation methods" className="span-2">
      <div className="row gap wrap">
        <Field label="Product profile"><ProfileSelect value={profile} onChange={setProfile} required /></Field>
        <Field label="Image"><input type="file" accept="image/*" onChange={(e) => setFile(e.target.files[0])} /></Field>
      </div>
      <div className="row gap wrap">
        {[["gradcam", "Grad-CAM"], ["gradcam_pp", "Grad-CAM++"], ["scorecam", "Score-CAM (slower)"]].map(([k, l]) => (
          <label key={k} className="check">
            <input type="checkbox" checked={methods[k]} onChange={(e) => setMethods({ ...methods, [k]: e.target.checked })} /> {l}
          </label>
        ))}
      </div>
      <p className="muted small">Class-activation maps apply to classifier profiles. Anomaly-only profiles (bottle, metal nut) return anomaly maps instead.</p>
      <ErrorBox error={error} />
      <button className="btn btn-primary" disabled={!file || !profile || busy} onClick={run}>{busy ? "Running…" : "Run and explain"}</button>
    </Card>
  );
}

function Analysis({ d }) {
  const r = d.result || {};
  const clf = r.engines?.classification;
  const an = r.engines?.anomaly;
  const nov = r.engines?.novelty;
  const keys = useMemo(() => Object.keys(d.visualizations || {}), [d]);
  const [left, setLeft] = useState("original");
  const [right, setRight] = useState("");
  useEffect(() => {
    setLeft(keys.includes("original") ? "original" : keys[0] || "");
    setRight(keys.find((k) => k.endsWith("_overlay") && k !== "original") || keys[1] || "");
  }, [keys]);

  const classes = clf ? Object.keys(clf.probabilities) : [];
  const probData = classes.map((c) => {
    const row = { cls: c, ensemble: clf.probabilities[c] };
    (clf.ensemble || []).forEach((m) => { row[m.arch] = m.probabilities[c]; });
    return row;
  });

  return (
    <div className="stack">
      <Card>
        <div className="row gap wrap" style={{ alignItems: "center" }}>
          <StatusBadge status={d.status} />
          <b>Inspection {short(d.id)}</b>
          <span className="muted">{d.profile_id} · {dt(d.created_at)}</span>
          <Link className="btn btn-ghost btn-sm" to={`/inspections/${d.id}`}>Full record →</Link>
        </div>
        <h4>Plain-language explanation</h4>
        <ul>
          {clf && <li>The classifier ensemble predicted <b>{clf.predicted_class}</b> with calibrated confidence <b>{pct(clf.confidence)}</b> (margin to the runner-up {pct(clf.margin)}).</li>}
          {clf && <li>Ensemble members {clf.members_agree ? "agree" : <b className="warn-text">disagree</b>} on the class{clf.members_agree ? "" : ", so the item is routed to manual review"}.</li>}
          {clf?.explanations && Object.entries(clf.explanations).map(([m, e]) => (
            <li key={m}>{LABELS[`${m}_overlay`]}: evidence peaks at {Math.round(e.peak_location_norm[0] * 100)}% across, {Math.round(e.peak_location_norm[1] * 100)}% down; {pct(e.activation_area_above_0_5 ?? e["activation_area_above_0.5"])} of the image is strongly activated.</li>
          ))}
          {an && <li>Anomaly detection: {an.is_anomalous ? <b>anomalous</b> : "within normal range"}. Primary engine {an.primary_engine} scored {num(an.score_ratio?.[an.primary_engine])}× its validated threshold{an.engines_disagree ? "; engines disagree → review" : ""}{an.within_review_margin ? "; score is inside the review margin → review" : ""}.</li>}
          {(nov || clf) && <li>Unfamiliar-input check: {d.unfamiliar ? <b className="warn-text">input looks unlike the training data — manual review required</b> : "input resembles the training data"}.</li>}
          <li>Decision: <b>{d.status}</b>{d.manual_review_required ? " — human review required" : ""}. Reliability: {d.reliability || "—"}.</li>
        </ul>
      </Card>

      {keys.length > 0 && (
        <Card title="Side-by-side viewer">
          <div className="grid grid-2">
            {[[left, setLeft], [right, setRight]].map(([k, set], i) => (
              <figure key={i} className="xai-pane">
                <select value={k} onChange={(e) => set(e.target.value)} aria-label={`Image ${i + 1}`}>
                  {keys.map((x) => <option key={x} value={x}>{LABELS[x] || x}</option>)}
                </select>
                {k && <img src={authUrl(d.visualizations[k])} alt={LABELS[k] || k} />}
                <figcaption className="muted small">{KIND_NOTE(k)}</figcaption>
              </figure>
            ))}
          </div>
        </Card>
      )}

      {clf && (
        <div className="grid grid-2">
          <Card title="Class probability distribution (calibrated)">
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={probData} margin={{ left: 0, right: 8 }}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} />
                <XAxis dataKey="cls" tick={{ fontSize: 11 }} interval={0} angle={-20} textAnchor="end" height={50} />
                <YAxis domain={[0, 1]} tickFormatter={(v) => `${Math.round(v * 100)}%`} width={40} />
                <Tooltip formatter={(v) => pct(v)} />
                <Legend />
                <Bar dataKey="ensemble" name="Ensemble" fill={PALETTE[0]} />
                {(clf.ensemble || []).map((m, i) => <Bar key={m.arch} dataKey={m.arch} fill={PALETTE[i + 1]} />)}
              </BarChart>
            </ResponsiveContainer>
            <p className="muted small">Entropy {num(clf.entropy, 3)} nats. A high softmax value is not proof of certainty: the out-of-distribution scores below are checked as well.</p>
          </Card>
          <Card title="Model agreement and out-of-distribution scores">
            <Table rows={clf.ensemble || []} rowKey={(m) => m.run_id} columns={[
              { label: "Model", render: (m) => m.arch },
              { label: "Top class", render: (m) => m.top_class === clf.predicted_class ? m.top_class : <b className="warn-text">{m.top_class}</b> },
              { label: "Conf.", render: (m) => pct(m.confidence) },
              { label: "Mahalanobis / thr.", render: (m) => <span className={m.ood ? "warn-text" : ""}>{num(m.mahalanobis, 1)} / {num(m.mahalanobis_threshold, 1)}</span> },
              { label: "Energy / thr.", render: (m) => `${num(m.energy, 2)} / ${num(m.energy_threshold, 2)}` },
              { label: "T", render: (m) => num(m.temperature, 2) },
            ]} />
            <p className="muted small">An item is flagged unfamiliar when a majority of members exceed their Mahalanobis threshold (99th percentile of validation features).</p>
          </Card>
        </div>
      )}

      {an && (
        <Card title="Anomaly scores against validated thresholds">
          <ResponsiveContainer width="100%" height={200}>
            <BarChart layout="vertical" data={Object.keys(an.scores).map((k) => ({ engine: k, ratio: an.score_ratio[k] }))} margin={{ left: 20, right: 20 }}>
              <CartesianGrid strokeDasharray="3 3" horizontal={false} />
              <XAxis type="number" domain={[0, (dataMax) => Math.max(2, Math.ceil(dataMax * 10) / 10)]} tickFormatter={(v) => `${v}×`} />
              <YAxis type="category" dataKey="engine" width={80} />
              <Tooltip formatter={(v) => `${num(v)}× threshold`} />
              <ReferenceLine x={1} stroke="#dc2626" label={{ value: "threshold", position: "top", fontSize: 11 }} />
              <ReferenceLine x={1 - an.review_margin} stroke="#d97706" strokeDasharray="4 4" />
              <ReferenceLine x={1 + an.review_margin} stroke="#d97706" strokeDasharray="4 4" />
              <Bar dataKey="ratio" fill={PALETTE[1]} />
            </BarChart>
          </ResponsiveContainer>
          <div className="row gap wrap small">
            {Object.entries(an.scores).map(([k, v]) => <Badge key={k}>{k}: {num(v, 3)} (thr. {num(an.thresholds[k], 3)})</Badge>)}
            <Badge>defect pixels {pct(an.defect_pixels_fraction, 2)}</Badge>
            {nov && <Badge kind={nov.unfamiliar ? "fail" : "muted"}>novelty {num(nov.score, 3)} / {num(nov.threshold, 3)}</Badge>}
          </div>
          <p className="muted small">Dashed lines: ±{pct(an.review_margin, 0)} review margin. {an.fusion}</p>
        </Card>
      )}
    </div>
  );
}
