import { Link } from "react-router-dom";
import { Async, Badge, Card, PageHeader, Stat, Table, bytes, num, useApi, useInterval } from "../components";

const ok = (v) => <Badge kind={v ? "pass" : "fail"}>{v ? "OK" : "problem"}</Badge>;

export default function SystemHealth() {
  const st = useApi("/api/system/status");
  useInterval(() => st.reload(), 10000);
  return (
    <>
      <PageHeader title="System health" subtitle="Live status of the API, database, models and host resources (refreshes every 10 s).">
        <button className="btn btn-ghost" onClick={st.reload}>Refresh</button>
      </PageHeader>
      <Async state={st}>
        {(s) => {
          const r = s.resources;
          const dev = s.device || {};
          return (
            <div className="stack">
              <div className="stats">
                <Stat label="Overall" value={s.status.toUpperCase()} kind={s.status === "ok" ? "pass" : "fail"} />
                <Stat label="API uptime" value={`${Math.round(s.api.uptime_s / 60)} min`} hint={s.api.environment} />
                <Stat label="Database" value={s.database.ok ? "OK" : "DOWN"} hint={`${s.database.dialect} · ${num(s.database.latency_ms, 1)} ms`} kind={s.database.ok ? "pass" : "fail"} />
                <Stat label="CPU" value={`${num(r.cpu_percent, 0)}%`} hint={`${r.cpu_count} logical CPUs`} />
                <Stat label="Memory" value={`${num(r.memory_percent, 0)}%`} hint={`${bytes(r.memory_used_bytes)} / ${bytes(r.memory_total_bytes)}`} kind={r.memory_percent > 90 ? "fail" : undefined} />
                <Stat label="GPU" value={dev.cuda_available ? "available" : "none"} hint={dev.device || ""} />
              </div>
              <div className="grid grid-2">
                <Card title="Model checkpoints used by active profiles" actions={<Link to="/models">Model registry →</Link>}>
                  <Table rows={s.models} rowKey={(m) => `${m.profile_id}-${m.run_id}`} empty="No models configured" columns={[
                    { label: "Profile", render: (m) => m.profile_id },
                    { label: "Kind", render: (m) => m.kind },
                    { label: "Run", render: (m) => <span className="small">{m.run_id}</span> },
                    { label: "On disk", render: (m) => ok(m.checkpoint_present) },
                    { label: "In memory", render: (m) => (m.loaded ? <Badge kind="info">loaded</Badge> : <span className="muted small">lazy</span>) },
                  ]} />
                  <p className="muted small">Models load on first use and stay cached. A missing checkpoint makes inspections for that profile fail with HTTP 503 — never a mock result.</p>
                </Card>
                <Card title="Process, device and storage">
                  <div className="kv">
                    <div><span className="muted">Server process memory</span><b>{bytes(r.process_rss_bytes)}</b></div>
                    <div><span className="muted">Server threads</span><b>{r.process_threads}</b></div>
                    <div><span className="muted">Torch</span><b>{dev.torch}</b></div>
                    <div><span className="muted">CPU threads (torch)</span><b>{dev.cpu_threads}</b></div>
                    <div><span className="muted">Schema mode</span><b>{s.database.schema_mode}</b></div>
                    <div><span className="muted">Stored files</span><b>{s.storage.files}</b></div>
                    <div><span className="muted">Storage used</span><b>{bytes(s.storage.bytes)}</b></div>
                    <div><span className="muted">Disk free</span><b>{bytes(s.storage.disk_free_bytes)} / {bytes(s.storage.disk_total_bytes)}</b></div>
                  </div>
                </Card>
              </div>
              <Card title="Integration adapters" actions={<Link to="/admin">Settings & API keys →</Link>}>
                <Table rows={s.integrations || []} rowKey={(a) => a.name} columns={[
                  { label: "Adapter", render: (a) => a.name },
                  { label: "State", render: (a) => <Badge kind={a.state === "active" ? "pass" : a.state === "disabled" || a.state === "not implemented" ? "muted" : "fail"}>{a.state}</Badge> },
                  { label: "Detail", render: (a) => <span className="small">{a.detail}</span> },
                  { label: "Delivered", render: (a) => a.stats ? JSON.stringify(a.stats) : "—" },
                ]} />
              </Card>
              <Card title={`Recent errors (${s.recent_errors.length}) · warnings since start: ${s.warning_count}`} actions={<Link to="/admin">Full logs & audit trail →</Link>}>
                <pre className="log">{s.recent_errors.map((e) => `${e.time.slice(0, 19)} ${e.level.padEnd(8)} ${e.logger}: ${e.message}`).join("\n") || "No errors logged since the server started."}</pre>
              </Card>
            </div>
          );
        }}
      </Async>
    </>
  );
}
