import { useState } from "react";
import { api, download } from "../api";
import { Async, Card, ErrorBox, Field, InspectionLink, PageHeader, ProfileSelect, Table, bytes, dt, useApi, useToast } from "../components";

const KINDS = {
  period: "Daily / weekly / monthly quality report",
  batch: "Batch inspection report",
  model: "Model evaluation report",
  comparison: "Model comparison report",
  dataset: "Dataset quality report",
  drift: "Drift monitoring report",
  review: "Human review report",
};

export default function Reports() {
  const list = useApi("/api/reports", { query: { limit: 200 } });
  const batches = useApi("/api/batches", { query: { limit: 100 } });
  const models = useApi("/api/models");
  const datasets = useApi("/api/datasets");
  const [f, setF] = useState({ kind: "period", period: "daily", profile_id: "", batch_id: "", model_id: "", dataset: "", days: 30, include_simulated: true });
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const toast = useToast();
  const set = (k) => (e) => setF({ ...f, [k]: e.target ? (e.target.type === "checkbox" ? e.target.checked : e.target.value) : e });

  const generate = async () => {
    setBusy(true);
    setError(null);
    try {
      const body = { kind: f.kind, period: f.period, profile_id: f.profile_id || null, include_simulated: f.include_simulated, days: +f.days };
      if (f.kind === "batch") body.batch_id = f.batch_id;
      if (f.kind === "model") body.model_id = +f.model_id;
      if (f.kind === "dataset") body.dataset = f.dataset;
      const r = await api("/api/reports/generate", { method: "POST", body });
      toast("Report generated", "success");
      list.reload();
      await download(r.download_url, `QualityGuard_${r.kind}_${r.id.slice(0, 8)}.pdf`);
    } catch (e) {
      setError(e);
    } finally {
      setBusy(false);
    }
  };

  return (
    <>
      <PageHeader title="Reports" subtitle="PDF reports with full traceability. Single-inspection reports are generated from each inspection page." />
      <div className="grid grid-3">
        <Card title="Generate">
          <Field label="Report type">
            <select value={f.kind} onChange={set("kind")}>
              {Object.entries(KINDS).map(([k, l]) => <option key={k} value={k}>{l}</option>)}
            </select>
          </Field>
          {f.kind === "period" && (
            <>
              <Field label="Period">
                <select value={f.period} onChange={set("period")}>
                  <option value="daily">Last 24 hours</option><option value="weekly">Last 7 days</option><option value="monthly">Last 30 days</option>
                </select>
              </Field>
              <Field label="Profile"><ProfileSelect value={f.profile_id} onChange={set("profile_id")} allowAll /></Field>
              <label className="check"><input type="checkbox" checked={f.include_simulated} onChange={set("include_simulated")} /> include simulated</label>
            </>
          )}
          {f.kind === "batch" && (
            <Field label="Batch">
              <select value={f.batch_id} onChange={set("batch_id")}>
                <option value="">Select…</option>
                {(batches.data || []).map((b) => <option key={b.id} value={b.id}>{b.name} ({b.inspections} items, {b.source})</option>)}
              </select>
            </Field>
          )}
          {f.kind === "model" && (
            <Field label="Model version">
              <select value={f.model_id} onChange={set("model_id")}>
                <option value="">Select…</option>
                {(models.data || []).map((m) => <option key={m.id} value={m.id}>{m.run_id}</option>)}
              </select>
            </Field>
          )}
          {f.kind === "comparison" && <p className="muted small">Compares all registered classification models on the shared held-out test split.</p>}
          {f.kind === "dataset" && (
            <Field label="Dataset">
              <select value={f.dataset} onChange={set("dataset")}>
                <option value="">Select…</option>
                {(datasets.data || []).filter((d) => d.versions.length).map((d) => <option key={d.name} value={d.name}>{d.title || d.name}</option>)}
              </select>
            </Field>
          )}
          {f.kind === "drift" && <Field label="Profile (latest check)"><ProfileSelect value={f.profile_id} onChange={set("profile_id")} allowAll /></Field>}
          {f.kind === "review" && <Field label="Days"><input type="number" min="1" max="365" value={f.days} onChange={set("days")} /></Field>}
          <button className="btn btn-primary btn-block" disabled={busy} onClick={generate}>{busy ? "Generating…" : "Generate & download"}</button>
          <ErrorBox error={error} />
        </Card>
        <Card title="Report archive" className="span-2">
          <Async state={list}>
            {(rows) => (
              <Table rows={rows} empty="No reports yet." columns={[
                { label: "Created", render: (r) => dt(r.created_at) },
                { label: "Type", render: (r) => r.kind },
                { label: "Title", key: "title" },
                { label: "Inspection", render: (r) => <InspectionLink id={r.inspection_id} /> },
                { label: "Size", render: (r) => bytes(r.size_bytes) },
                { label: "", render: (r) => <button className="btn btn-ghost btn-sm" onClick={() => download(r.download_url, `QualityGuard_${r.kind}_${r.id.slice(0, 8)}.${r.format}`).catch((e) => toast(e.message, "error"))}>Download</button> },
              ]} />
            )}
          </Async>
        </Card>
      </div>
    </>
  );
}
