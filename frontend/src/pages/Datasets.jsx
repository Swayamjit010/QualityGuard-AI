import { useState } from "react";
import { Link, Route, Routes, useNavigate, useParams } from "react-router-dom";
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api, authUrl } from "../api";
import {
  Async, Badge, Card, ErrorBox, Field, JsonView, Modal, PALETTE, PageHeader, Pager, Table, can, dt, useApi, useAuth, useToast,
} from "../components";

export default function Datasets() {
  return (
    <Routes>
      <Route index element={<DatasetList />} />
      <Route path=":name" element={<DatasetDetail />} />
    </Routes>
  );
}

function DatasetList() {
  const state = useApi("/api/datasets");
  const { user } = useAuth();
  const navigate = useNavigate();
  const toast = useToast();
  const [importing, setImporting] = useState(false);
  const exportFeedback = async () => {
    try {
      const r = await api("/api/datasets/feedback-export", { method: "POST" });
      toast(r.exported ? `Exported ${r.exported} reviewed images to ${r.path}` : "No new reviewed inspections to export", "success");
    } catch (e) {
      toast(e.message, "error");
    }
  };
  return (
    <>
      <PageHeader title="Datasets" subtitle="Versioned datasets with leakage-safe splits (duplicates grouped into one split).">
        {can(user, "export") && <button className="btn btn-ghost" onClick={exportFeedback}>Export reviewed feedback</button>}
        {can(user, "manage_datasets") && <button className="btn btn-primary" onClick={() => setImporting(true)}>Import ZIP</button>}
      </PageHeader>
      <Card>
        <Async state={state}>
          {(rows) => (
            <Table rows={rows} onRowClick={(d) => navigate(`/datasets/${d.name}`)} columns={[
              { label: "Name", render: (d) => <b>{d.name}</b> },
              { label: "Title", key: "title" },
              { label: "Task", key: "task" },
              { label: "Versions", render: (d) => d.versions.length },
              { label: "Samples", render: (d) => d.versions.at(-1)?.num_samples ?? "—" },
              { label: "Split (train/val/test)", render: (d) => { const s = d.versions.at(-1)?.split_counts; return s ? `${s.train ?? 0} / ${s.val ?? 0} / ${s.test ?? 0}` : "—"; } },
              { label: "License", render: (d) => <span className="small">{d.license}</span> },
            ]} />
          )}
        </Async>
      </Card>
      {importing && <ImportModal onClose={() => setImporting(false)} onDone={() => { setImporting(false); state.reload(); }} />}
    </>
  );
}

function ImportModal({ onClose, onDone }) {
  const [file, setFile] = useState(null);
  const [name, setName] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const go = async () => {
    setBusy(true);
    setError(null);
    const fd = new FormData();
    fd.append("file", file);
    fd.append("name", name);
    try {
      await api("/api/datasets/import", { method: "POST", form: fd });
      onDone();
    } catch (e) {
      setError(e);
    } finally {
      setBusy(false);
    }
  };
  return (
    <Modal title="Import dataset" onClose={onClose}>
      <p className="muted small">ZIP with one folder per class (<code>scratch/001.png</code>…) or images plus a <code>labels.csv</code> (file,label).</p>
      <Field label="Dataset name" hint="lowercase letters, digits and _"><input value={name} onChange={(e) => setName(e.target.value)} pattern="[a-z0-9_]{2,64}" /></Field>
      <Field label="Archive"><input type="file" accept=".zip" onChange={(e) => setFile(e.target.files[0])} /></Field>
      <ErrorBox error={error} />
      <button className="btn btn-primary" disabled={!file || !name || busy} onClick={go}>{busy ? "Importing…" : "Import"}</button>
    </Modal>
  );
}

function DatasetDetail() {
  const { name } = useParams();
  const state = useApi(`/api/datasets/${name}`);
  const { user } = useAuth();
  const toast = useToast();
  const [split, setSplit] = useState("");
  const [label, setLabel] = useState("");
  const [page, setPage] = useState(1);
  const [showStats, setShowStats] = useState(false);
  const samples = useApi(state.data?.versions.length ? `/api/datasets/${name}/samples` : null, { query: { split, label, page, page_size: 36 } });

  const analyze = async () => {
    try {
      const r = await api(`/api/datasets/${name}/analyze`, { method: "POST", body: { train: 0.7, val: 0.15, test: 0.15, seed: 42 } });
      toast(`Version ${r.version}: ${r.num_samples} samples`, "success");
      state.reload();
    } catch (e) {
      toast(e.message, "error");
    }
  };

  return (
    <Async state={state}>
      {(d) => {
        const st = d.statistics || {};
        const classDist = st.class_distribution ? Object.entries(st.class_distribution).map(([k, v]) => ({ name: k, count: v })) : [];
        return (
          <>
            <PageHeader title={d.title || d.name} subtitle={`${d.task} · ${d.layout} · ${d.source}`}>
              {can(user, "manage_datasets") && <button className="btn btn-ghost" onClick={analyze}>Re-analyse & split</button>}
              {d.statistics && <button className="btn btn-ghost" onClick={() => setShowStats(true)}>Statistics JSON</button>}
              <Link className="btn btn-ghost" to="/datasets">← Datasets</Link>
            </PageHeader>
            <div className="grid grid-3">
              <Card title="Versions">
                <Table rows={d.versions} columns={[
                  { label: "Version", render: (v) => <span className="mono small">{v.version}</span> },
                  { label: "Samples", key: "num_samples" },
                  { label: "Split", render: (v) => Object.entries(v.split_counts).map(([k, n]) => `${k}:${n}`).join(" ") },
                  { label: "Created", render: (v) => dt(v.created_at) },
                ]} />
                <p className="muted small">{d.description}</p>
                <p className="small">License: {d.license}</p>
              </Card>
              <Card title="Class balance" className="span-2">
                {classDist.length ? (
                  <ResponsiveContainer width="100%" height={240}>
                    <BarChart data={classDist}>
                      <CartesianGrid strokeDasharray="3 3" stroke="var(--grid)" />
                      <XAxis dataKey="name" fontSize={11} />
                      <YAxis allowDecimals={false} fontSize={11} />
                      <Tooltip />
                      <Bar dataKey="count" fill={PALETTE[0]} radius={[4, 4, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                ) : <div className="empty">Run analysis to compute statistics.</div>}
                <div className="row gap wrap small">
                  {st.corrupted && <Badge kind={st.corrupted.length ? "fail" : "pass"}>{st.corrupted.length} corrupted</Badge>}
                  {st.exact_duplicates && <Badge kind={st.exact_duplicates.length ? "review" : "pass"}>{st.exact_duplicates.length} exact-duplicate groups</Badge>}
                  {st.near_duplicates && <Badge kind={st.near_duplicates.length ? "review" : "pass"}>{st.near_duplicates.length} near-duplicate pairs</Badge>}
                  {st.imbalance_ratio != null && <Badge kind={st.imbalanced ? "review" : "pass"}>imbalance {Number(st.imbalance_ratio).toFixed(2)}×</Badge>}
                  {st.num_quality_flagged != null && <Badge kind={st.num_quality_flagged ? "review" : "pass"}>{st.num_quality_flagged} quality-flagged</Badge>}
                  {st.image_sizes && <Badge>sizes: {Object.keys(st.image_sizes).slice(0, 3).join(", ")}</Badge>}
                </div>
              </Card>
            </div>
            <Card title="Samples" actions={
              <div className="row gap-s">
                <select value={split} onChange={(e) => { setSplit(e.target.value); setPage(1); }}>
                  <option value="">All splits</option><option value="train">train</option><option value="val">val</option><option value="test">test</option>
                </select>
                <select value={label} onChange={(e) => { setLabel(e.target.value); setPage(1); }}>
                  <option value="">All labels</option>
                  {(samples.data?.labels || []).map((l) => <option key={l}>{l}</option>)}
                </select>
              </div>
            }>
              <Async state={samples}>
                {(s) => (
                  <>
                    <div className="gallery gallery-sm">
                      {s.items.map((x) => (
                        <figure key={x.index}>
                          <img src={authUrl(x.url)} alt={x.file} loading="lazy" />
                          <figcaption><Badge>{x.split}</Badge> {x.label}</figcaption>
                        </figure>
                      ))}
                    </div>
                    <Pager page={page} pageSize={36} total={s.total} onPage={setPage} />
                  </>
                )}
              </Async>
            </Card>
            {showStats && <Modal title="Dataset statistics" wide onClose={() => setShowStats(false)}><JsonView value={d.statistics} /></Modal>}
          </>
        );
      }}
    </Async>
  );
}
