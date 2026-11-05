import { useState } from "react";
import { Link, useParams } from "react-router-dom";
import { api, authUrl, download } from "../api";
import {
  Async, Badge, Card, ErrorBox, Field, JsonView, Modal, PageHeader, SeverityBadge, StatusBadge, Table, can, dt, num,
  pct, short, useApi, useAuth, useToast,
} from "../components";

export const PLAIN_VERDICT = {
  PASS: "✓ NOT DEFECTIVE", FAIL: "✗ DEFECTIVE", "REVIEW REQUIRED": "? NEEDS HUMAN REVIEW", INCONCLUSIVE: "– CAN'T DECIDE",
};

const VIS_LABEL = {
  original: "Input", gradcam_overlay: "Grad-CAM", gradcam_pp_overlay: "Grad-CAM++", scorecam_overlay: "Score-CAM",
  anomaly_patchcore_overlay: "PatchCore anomaly map", anomaly_padim_overlay: "PaDiM anomaly map",
  segmentation_overlay: "Estimated defect region", segmentation_mask: "Segmentation mask", bounding_boxes: "Regions",
  defect_zoom: "Largest region (zoom)", comparison: "Original vs. result",
};

export function ResultPanel({ d, compact }) {
  const r = d.result || {};
  const dec = r.decision || {};
  const clf = r.engines?.classification;
  const an = r.engines?.anomaly;
  const nov = r.engines?.novelty;
  const m = r.measurements;
  const [zoom, setZoom] = useState(null);
  const vis = Object.entries(d.visualizations || {}).filter(([k]) => !k.endsWith("_heatmap"));

  return (
    <div className="stack">
      <Card>
        <div className={`verdict verdict-${(d.status || "").replace(" ", "-")}`}>
          <div>
            <div className="verdict-status" data-testid="detail-verdict">{d.verdict || PLAIN_VERDICT[d.status] || d.status}</div>
            <div className="verdict-code">Line disposition: {d.status}{d.human_decision ? ` · human decision: ${d.human_decision}` : ""}</div>
            {r.recognition?.block && <div className="verdict-msg"><b>⚠ {r.recognition.block}</b></div>}
            {r.recognition?.title && !r.recognition?.block && (
              <div className="verdict-code">Recognised category: {r.recognition.title} ({pct(r.recognition.probability)})</div>
            )}
            <div className="verdict-msg">{dec.message || dec.recommendation}</div>
          </div>
          <div className="verdict-qi">
            <div className="muted small">Quality index</div>
            <div className="verdict-qi-val">{num(d.quality_index, 0)}</div>
          </div>
        </div>
        <div className="kv kv-4">
          <div><span className="muted">Defect</span><b>{d.defect_class || (d.defect_present ? "anomaly (unclassified)" : "none")}</b></div>
          <div><span className="muted">Severity</span><SeverityBadge level={d.severity} /></div>
          <div><span className="muted">Confidence</span><b>{pct(d.confidence)}</b></div>
          <div><span className="muted">Anomaly / threshold</span><b>{d.anomaly_score_ratio != null ? `${num(d.anomaly_score_ratio)}×` : "—"}</b></div>
          <div><span className="muted">Reliability</span><b>{d.reliability || "—"}</b></div>
          <div><span className="muted">Unfamiliar input</span><b>{d.unfamiliar ? "yes" : "no"}</b></div>
          <div><span className="muted">Review</span><b>{d.review_status}</b></div>
          <div><span className="muted">Latency</span><b>{num(d.latency_ms, 0)} ms</b></div>
        </div>
        {compact && <Link className="btn btn-ghost btn-sm" to={`/inspections/${d.id}`}>Open full record · review · report →</Link>}
      </Card>

      {vis.length > 0 && (
        <Card title="Visual explanation">
          <div className="gallery">
            {vis.map(([k, url]) => (
              <figure key={k} onClick={() => setZoom([k, url])}>
                <img src={authUrl(url)} alt={k} loading="lazy" />
                <figcaption>{VIS_LABEL[k] || k.replaceAll("_", " ")}</figcaption>
              </figure>
            ))}
          </div>
          <p className="muted small">Grad-CAM shows where the classifier looked — it is an explanation, not a defect boundary. Anomaly maps show deviation from defect-free references.</p>
        </Card>
      )}
      {zoom && (
        <Modal title={VIS_LABEL[zoom[0]] || zoom[0]} onClose={() => setZoom(null)} wide>
          <img className="zoom-img" src={authUrl(zoom[1])} alt={zoom[0]} />
        </Modal>
      )}

      <div className="grid grid-2">
        <Card title="Decision basis">
          {(dec.reasons || []).length > 0 && <><h4>Reasons</h4><ul>{dec.reasons.map((x, i) => <li key={i}>{x}</li>)}</ul></>}
          {(r.severity?.basis || []).length > 0 && <><h4>Severity basis</h4><ul>{r.severity.basis.map((x, i) => <li key={i}>{x}</li>)}</ul></>}
          {(r.reliability?.notes || []).length > 0 && <><h4>Reliability</h4><ul>{r.reliability.notes.map((x, i) => <li key={i}>{x}</li>)}</ul></>}
          {(r.input_issues || []).length > 0 && <><h4>Input issues</h4><ul>{r.input_issues.map((x, i) => <li key={i}>{x}</li>)}</ul></>}
          {(dec.rules_applied || []).length > 0 && <><h4>Rules applied</h4><ul className="small">{dec.rules_applied.map((x, i) => <li key={i}>{typeof x === "string" ? x : JSON.stringify(x)}</li>)}</ul></>}
        </Card>
        <Card title="Model outputs">
          {clf && (
            <>
              <h4>Classification ({clf.ensemble?.length} model{clf.ensemble?.length > 1 ? "s" : ""})</h4>
              {Object.entries(clf.probabilities || {}).sort((a, b) => b[1] - a[1]).map(([k, v]) => (
                <div key={k} className="prob-row">
                  <span>{k}</span>
                  <div className="prob-bar"><div style={{ width: `${v * 100}%` }} /></div>
                  <span className="mono small">{pct(v)}</span>
                </div>
              ))}
              {clf.ensemble?.length > 1 && (
                <Table rows={clf.ensemble} rowKey={(x) => x.run_id} columns={[
                  { label: "Model", render: (x) => <span className="mono small">{x.run_id}</span> },
                  { label: "Top", key: "top_class" },
                  { label: "Conf.", render: (x) => pct(x.confidence) },
                  { label: "OOD", render: (x) => (x.ood ? <Badge kind="review">yes</Badge> : "no") },
                ]} />
              )}
            </>
          )}
          {an && (
            <>
              <h4>Anomaly detection <span className="muted small">({an.model})</span></h4>
              <Table rows={Object.keys(an.scores)} rowKey={(k) => k} columns={[
                { label: "Engine", render: (k) => k + (k === an.primary_engine ? " (primary)" : "") },
                { label: "Score", render: (k) => num(an.scores[k], 3) },
                { label: "Threshold", render: (k) => num(an.thresholds[k], 3) },
                { label: "Ratio", render: (k) => `${num(an.score_ratio[k])}×` },
              ]} />
              {an.engines_disagree && <p className="warn-text">Engines disagree → manual review.</p>}
            </>
          )}
          {nov && <p className="small">Novelty score {num(nov.score, 3)} / threshold {num(nov.threshold, 3)} → {nov.unfamiliar ? "unfamiliar" : "familiar"}</p>}
        </Card>
      </div>

      {m && (
        <Card title={`Defect regions (${m.num_regions})`}>
          <div className="kv kv-4">
            <div><span className="muted">Defect area</span><b>{pct(m.defect_area_ratio, 2)}</b></div>
            <div><span className="muted">Total area</span><b>{m.calibrated ? `${num(m.total_defect_area_mm2)} mm²` : `${num(m.total_defect_area_px, 0)} px`}</b></div>
            <div><span className="muted">Largest region</span><b>{num(m.largest_region_px, 0)} px</b></div>
            <div><span className="muted">Spread</span><b>{num(m.spatial_spread, 3)}</b></div>
          </div>
          <Table rows={m.regions || []} rowKey={(x) => x.id} columns={[
            { label: "#", key: "id" },
            { label: "Area (px)", render: (x) => num(x.area_px, 0) },
            { label: "BBox (x,y,w,h)", render: (x) => (x.bbox_px || []).join(", ") },
            { label: "Peak score", render: (x) => num(x.peak_score ?? x.max_score, 3) },
          ]} />
          {!m.calibrated && <p className="muted small">Not calibrated: sizes are in pixels (set calibration.mm_per_pixel in the profile).</p>}
        </Card>
      )}
    </div>
  );
}

function FeedbackForm({ d, onDone }) {
  const profile = useApi(`/api/profiles/${d.profile_id}`);
  const [f, setF] = useState({ verdict: "confirm", final_decision: d.status === "PASS" ? "accept" : "reject", corrected_class: "",
    corrected_severity: "", location_confirmed: null, difficult: false, comment: "" });
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);
  const set = (k) => (e) => setF({ ...f, [k]: e.target.type === "checkbox" ? e.target.checked : e.target.value });
  const submit = async (e) => {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      await api(`/api/inspections/${d.id}/feedback`, { method: "POST", body: {
        ...f, corrected_class: f.corrected_class || null, corrected_severity: f.corrected_severity || null,
        location_confirmed: f.location_confirmed === "" ? null : f.location_confirmed } });
      onDone();
    } catch (err) {
      setError(err);
    } finally {
      setBusy(false);
    }
  };
  const classes = [...(profile.data?.defect_classes || []), ...(profile.data?.config?.normal_classes || [])];
  return (
    <form onSubmit={submit} className="stack-s">
      <div className="row gap wrap">
        <Field label="AI result">
          <select value={f.verdict} onChange={set("verdict")}>
            <option value="confirm">Confirm (AI was right)</option>
            <option value="reject">Reject (AI was wrong)</option>
          </select>
        </Field>
        <Field label="Final decision">
          <select value={f.final_decision} onChange={set("final_decision")}>
            <option value="accept">Accept item</option>
            <option value="reject">Reject item</option>
          </select>
        </Field>
        <Field label="Correct class">
          <select value={f.corrected_class} onChange={set("corrected_class")}>
            <option value="">(unchanged)</option>
            {classes.map((c) => <option key={c} value={c}>{c}</option>)}
          </select>
        </Field>
        <Field label="Correct severity">
          <select value={f.corrected_severity} onChange={set("corrected_severity")}>
            <option value="">(unchanged)</option>
            {["none", "low", "medium", "high", "critical"].map((c) => <option key={c} value={c}>{c}</option>)}
          </select>
        </Field>
        <Field label="Location correct?">
          <select value={String(f.location_confirmed)} onChange={(e) => setF({ ...f, location_confirmed: e.target.value === "null" ? null : e.target.value === "true" })}>
            <option value="null">n/a</option><option value="true">yes</option><option value="false">no</option>
          </select>
        </Field>
      </div>
      <label className="check"><input type="checkbox" checked={f.difficult} onChange={set("difficult")} /> Difficult case (prioritise for retraining)</label>
      <Field label="Comment"><textarea rows={3} maxLength={4000} value={f.comment} onChange={set("comment")} /></Field>
      <ErrorBox error={error} />
      <button className="btn btn-primary" disabled={busy}>Submit review</button>
    </form>
  );
}

function Annotations({ d, canEdit, onChange }) {
  const [kind, setKind] = useState("mask");
  const [file, setFile] = useState(null);
  const [boxes, setBoxes] = useState("[[10, 10, 50, 50]]");
  const [label, setLabel] = useState("");
  const [error, setError] = useState(null);
  const toast = useToast();
  const anns = d.annotations || [];
  const w = d.result?.input?.width;
  const h = d.result?.input?.height;
  const save = async () => {
    setError(null);
    const fd = new FormData();
    fd.append("kind", kind);
    if (label) fd.append("label", label);
    try {
      if (kind === "mask") fd.append("file", file);
      if (kind === "bbox") fd.append("data", JSON.stringify({ boxes: JSON.parse(boxes) }));
      await api(`/api/inspections/${d.id}/annotations`, { method: "POST", form: fd });
      toast("Annotation saved", "success");
      setFile(null);
      onChange();
    } catch (e) {
      setError(e);
    }
  };
  const remove = async (a) => {
    try {
      await api(`/api/inspections/${d.id}/annotations/${a.id}`, { method: "DELETE" });
      onChange();
    } catch (e) {
      toast(e.message, "error");
    }
  };
  if (!canEdit && anns.length === 0) return null;
  return (
    <Card title={`Inspector annotations (${anns.length})`}>
      {anns.length > 0 && (
        <ul className="plain">
          {anns.map((a) => (
            <li key={a.id} className="feedback-item row gap-s" style={{ alignItems: "center" }}>
              {a.mask_url && <img src={authUrl(a.mask_url)} alt="annotation mask" style={{ width: 56, height: 56, objectFit: "contain", background: "#000" }} />}
              <div className="small">
                <b>{a.kind}</b>{a.label && <> · {a.label}</>} · {dt(a.created_at)}
                {a.kind === "mask" && <> · {pct(a.data.defect_fraction, 2)} of image</>}
                {a.kind === "bbox" && <> · {a.data.boxes.length} box(es)</>}
              </div>
              {canEdit && <button className="btn btn-ghost btn-sm" onClick={() => remove(a)}>Remove</button>}
            </li>
          ))}
        </ul>
      )}
      {canEdit && (
        <div className="stack-s">
          <div className="row gap-s wrap">
            <select value={kind} onChange={(e) => setKind(e.target.value)} aria-label="Annotation type">
              <option value="mask">Defect mask (PNG)</option>
              <option value="bbox">Bounding boxes</option>
              <option value="label">Label only</option>
            </select>
            <select value={label} onChange={(e) => setLabel(e.target.value)} aria-label="Annotation label">
              <option value="">(no label)</option>
              {(d.result?.engines?.classification ? Object.keys(d.result.engines.classification.probabilities) : ["good"]).map((c) => <option key={c}>{c}</option>)}
            </select>
          </div>
          {kind === "mask" && <input type="file" accept="image/png" onChange={(e) => setFile(e.target.files[0])} aria-label="Mask PNG" />}
          {kind === "bbox" && <Field label="Boxes [[x, y, width, height], …]" hint={w ? `original image is ${w}×${h} px` : ""}><input value={boxes} onChange={(e) => setBoxes(e.target.value)} className="mono" /></Field>}
          <p className="muted small">Masks: white (non-zero) = defect, same aspect ratio as the image. Annotations are kept with the image for curated retraining datasets.</p>
          <ErrorBox error={error} />
          <button className="btn btn-ghost" disabled={kind === "mask" && !file} onClick={save}>Add annotation</button>
        </div>
      )}
    </Card>
  );
}

export default function InspectionDetail() {
  const { id } = useParams();
  const { user, refreshCounts } = useAuth();
  const toast = useToast();
  const state = useApi(`/api/inspections/${id}`);
  const [similar, setSimilar] = useState(null);
  const [showRaw, setShowRaw] = useState(false);
  const kb = useApi(state.data?.defect_class ? "/api/knowledge" : null, { query: { defect_class: state.data?.defect_class } });

  const report = async () => {
    try {
      await download(`/api/inspections/${id}/report`, `QualityGuard_inspection_${short(id)}.pdf`);
    } catch (e) {
      toast(e.message, "error");
    }
  };
  const loadSimilar = async () => setSimilar(await api(`/api/inspections/${id}/similar`, { query: { k: 8 } }));

  return (
    <Async state={state}>
      {(d) => (
        <>
          <PageHeader title={<>Inspection <span className="mono">{short(d.id)}</span></>}
                      subtitle={`${dt(d.created_at)} · profile ${d.profile_id} v${d.traceability.profile_version} · ${d.source}${d.is_simulated ? " · SIMULATED" : ""}`}>
            <StatusBadge status={d.status} />
            {can(user, "reports") && <button className="btn btn-primary" onClick={report}>PDF report</button>}
            <button className="btn btn-ghost" onClick={loadSimilar}>Find similar</button>
            <Link className="btn btn-ghost" to={`/xai?inspection=${d.id}`}>Open in XAI lab</Link>
            <button className="btn btn-ghost" onClick={() => setShowRaw(true)}>Raw JSON</button>
          </PageHeader>
          <div className="grid grid-3">
            <div className="span-2"><ResultPanel d={d} /></div>
            <div className="stack">
              <Card title="Human review">
                {d.feedback.length > 0 && (
                  <ul className="plain">
                    {d.feedback.map((f) => (
                      <li key={f.id} className="feedback-item">
                        <div><b>{f.reviewer}</b> · {dt(f.created_at)}</div>
                        <div>{f.verdict === "confirm" ? "Confirmed" : "Rejected"} AI result → <b>{f.final_decision}</b>
                          {f.corrected_class && <> · class: {f.corrected_class}</>}
                          {f.corrected_severity && <> · severity: {f.corrected_severity}</>}
                          {f.difficult && <> · <Badge kind="review">difficult</Badge></>}</div>
                        {f.comment && <div className="muted">“{f.comment}”</div>}
                      </li>
                    ))}
                  </ul>
                )}
                {can(user, "review") ? (
                  <FeedbackForm d={d} onDone={() => { toast("Review saved", "success"); state.reload(); refreshCounts(); }} />
                ) : d.feedback.length === 0 && <div className="empty">No review yet.</div>}
              </Card>
              <Annotations d={d} canEdit={can(user, "review")} onChange={state.reload} />
              <Card title="Traceability">
                <div className="kv">
                  <div><span className="muted">File</span><b>{d.filename}</b></div>
                  <div><span className="muted">Serial</span><b>{d.product_serial || "—"}</b></div>
                  <div><span className="muted">Batch</span><b>{d.batch_id ? <Link to={`/inspections?batch_id=${d.batch_id}`}>{short(d.batch_id)}</Link> : "—"}</b></div>
                  <div><span className="muted">Models</span><b className="mono small">{(d.traceability.models || []).join(", ") || "—"}</b></div>
                  <div><span className="muted">Ground truth</span><b>{d.ground_truth || "—"}</b></div>
                  <div><span className="muted">Image</span><b><a href={authUrl(`/api/inspections/${d.id}/images/upload`)} target="_blank" rel="noreferrer">original upload</a></b></div>
                </div>
              </Card>
              {kb.data?.length > 0 && (
                <Card title={`Knowledge: ${d.defect_class}`}>
                  {kb.data.slice(0, 1).map((k) => (
                    <div key={k.id} className="small">
                      <p>{k.visual_description}</p>
                      {k.inspection_guidance?.length > 0 && <><h4>Inspection guidance</h4><ul>{k.inspection_guidance.map((x, i) => <li key={i}>{x}</li>)}</ul></>}
                      {k.corrective_actions?.length > 0 && <><h4>Possible corrective actions</h4><ul>{k.corrective_actions.map((x, i) => <li key={i}>{x.text || String(x)} {x.verified ? <Badge kind="pass">verified</Badge> : <Badge>unverified</Badge>}</li>)}</ul></>}
                    </div>
                  ))}
                </Card>
              )}
            </div>
          </div>
          {similar && (
            <Card title="Visually similar past inspections" actions={<span className="muted small">{similar.caveat}</span>}>
              {similar.results.length ? (
                <div className="gallery">
                  {similar.results.map((s) => (
                    <Link key={s.id} to={`/inspections/${s.id}`} onClick={() => setSimilar(null)}>
                      <figure>
                        <img src={authUrl(`/api/inspections/${s.id}/images/original`)} alt={s.id} loading="lazy" />
                        <figcaption><StatusBadge status={s.status} /> {s.defect_class || ""} · sim {num(s.similarity, 2)}</figcaption>
                      </figure>
                    </Link>
                  ))}
                </div>
              ) : <div className="empty">No comparable inspections (same embedding model) yet.</div>}
            </Card>
          )}
          {showRaw && <Modal title="Raw result" onClose={() => setShowRaw(false)} wide><JsonView value={d.result} /></Modal>}
        </>
      )}
    </Async>
  );
}
