import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import {
  Bar, BarChart, CartesianGrid, ComposedChart, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import { api } from "../api";
import { Async, Card, Empty, PALETTE, PageHeader, ProfileSelect, STATUS_COLORS, Stat, Table, num, pct, useApi, useToast } from "../components";

export default function Analytics() {
  const [days, setDays] = useState(30);
  const [profile, setProfile] = useState("");
  const [unit, setUnit] = useState("day");
  const [sim, setSim] = useState(true);
  const q = { days, profile_id: profile, include_simulated: sim };
  const trend = useApi("/api/analytics/trend", { query: { ...q, unit } });
  const defects = useApi("/api/analytics/defects", { query: q });
  const quality = useApi("/api/analytics/quality", { query: q });
  const forecast = useApi("/api/analytics/forecast", { query: { ...q, days: Math.max(days, 30) } });
  const models = useApi("/api/analytics/models", { query: q });
  const drift = useApi("/api/drift/checks", { query: { profile_id: profile, limit: 10 } });
  const toast = useToast();
  const navigate = useNavigate();

  const runDrift = async () => {
    if (!profile) return toast("Select a profile to run a drift check", "warn");
    try {
      const r = await api("/api/drift/check", { method: "POST", query: { profile_id: profile } });
      toast(r.details.interpretation, r.data_drift || r.prediction_drift ? "warn" : "success");
      drift.reload();
    } catch (e) {
      toast(e.message, "error");
    }
  };

  return (
    <>
      <PageHeader title="Analytics" subtitle="Every figure is a database aggregate over recorded inspections.">
        <ProfileSelect value={profile} onChange={setProfile} allowAll />
        <select value={days} onChange={(e) => setDays(+e.target.value)}>
          {[7, 30, 90, 365].map((d) => <option key={d} value={d}>Last {d} days</option>)}
        </select>
        <select value={unit} onChange={(e) => setUnit(e.target.value)}>
          {["hour", "day", "week", "month"].map((u) => <option key={u} value={u}>by {u}</option>)}
        </select>
        <label className="check"><input type="checkbox" checked={sim} onChange={(e) => setSim(e.target.checked)} /> include simulated</label>
      </PageHeader>

      <Async state={quality}>
        {(qd) => (
          <div className="stats">
            <Stat label="Mean quality index" value={num(qd.quality_index_mean, 1)} />
            <Stat label="Latency p50 / p95" value={qd.latency_ms ? `${num(qd.latency_ms.p50, 0)} / ${num(qd.latency_ms.p95, 0)} ms` : "—"} />
            <Stat label="Throughput (24h)" value={`${num(qd.throughput.per_hour_last_24h, 1)}/h`} hint={`peak ${qd.throughput.max_per_minute}/min`} />
            <Stat label="False-negative rate" value={pct(qd.ground_truth.false_negative_rate)} kind="fail" hint={`${qd.ground_truth.evaluated} with ground truth`} />
            <Stat label="False-positive rate" value={pct(qd.ground_truth.false_positive_rate)} kind="review" hint={`class acc ${pct(qd.ground_truth.class_accuracy)}`} />
          </div>
        )}
      </Async>

      <div className="grid grid-2">
        <Card title="Outcomes over time" actions={<Link to={`/inspections${profile ? `?profile_id=${profile}` : ""}`}>All inspections →</Link>}>
          <Async state={trend}>
            {(t) => t.length ? (
              <ResponsiveContainer width="100%" height={260}>
                <ComposedChart data={t}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--grid)" />
                  <XAxis dataKey="period" fontSize={11} />
                  <YAxis yAxisId="l" allowDecimals={false} fontSize={11} />
                  <YAxis yAxisId="r" orientation="right" domain={[0, 1]} tickFormatter={(v) => `${Math.round(v * 100)}%`} fontSize={11} />
                  <Tooltip formatter={(v, n) => (n === "defect rate" ? pct(v) : v)} />
                  <Legend />
                  <Bar yAxisId="l" dataKey="pass" stackId="s" name="PASS" fill={STATUS_COLORS.PASS} />
                  <Bar yAxisId="l" dataKey="fail" stackId="s" name="FAIL" fill={STATUS_COLORS.FAIL} />
                  <Bar yAxisId="l" dataKey="review" stackId="s" name="REVIEW" fill={STATUS_COLORS["REVIEW REQUIRED"]} />
                  <Bar yAxisId="l" dataKey="inconclusive" stackId="s" name="INCONCLUSIVE" fill={STATUS_COLORS.INCONCLUSIVE} />
                  <Line yAxisId="r" dataKey="defect_rate" name="defect rate" stroke="#0f172a" dot={false} />
                </ComposedChart>
              </ResponsiveContainer>
            ) : <Empty>No data.</Empty>}
          </Async>
        </Card>
        <Card title="Defect-rate forecast">
          <Async state={forecast}>
            {(f) => f.available ? (
              <>
                <ResponsiveContainer width="100%" height={230}>
                  <LineChart data={[...f.history.map((h) => ({ ...h, hist: h.defect_rate })), ...f.forecast.map((h) => ({ ...h, fc: h.defect_rate }))]}>
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--grid)" />
                    <XAxis dataKey="period" fontSize={11} />
                    <YAxis tickFormatter={(v) => `${Math.round(v * 100)}%`} fontSize={11} />
                    <Tooltip formatter={(v) => pct(v)} />
                    <Line dataKey="hist" name="history" stroke="#1d6fe0" dot={false} />
                    <Line dataKey="fc" name="forecast" stroke="#f59e0b" strokeDasharray="5 4" dot={false} />
                  </LineChart>
                </ResponsiveContainer>
                <p className="muted small">{f.method}. Backtest MAE {num(f.backtest_mae, 3)} vs naive {num(f.naive_mae, 3)} ({f.beats_naive ? "beats" : "does not beat"} naive). {f.caveat}</p>
              </>
            ) : <Empty>{f.reason}</Empty>}
          </Async>
        </Card>
        <Card title="Defects by class" actions={<span className="muted small">click a bar to see those inspections</span>}>
          <Async state={defects}>
            {(d) => d.by_class.length ? (
              <ResponsiveContainer width="100%" height={240}>
                <BarChart data={d.by_class}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--grid)" />
                  <XAxis dataKey="name" fontSize={11} />
                  <YAxis allowDecimals={false} fontSize={11} />
                  <Tooltip />
                  <Bar dataKey="count" fill={PALETTE[0]} radius={[4, 4, 0, 0]} cursor="pointer"
                       onClick={(b) => navigate(b.name === "anomaly" ? `/inspections?status=FAIL${profile ? `&profile_id=${profile}` : ""}` : `/inspections?defect_class=${encodeURIComponent(b.name)}${profile ? `&profile_id=${profile}` : ""}`)} />
                </BarChart>
              </ResponsiveContainer>
            ) : <Empty>No defects recorded.</Empty>}
          </Async>
        </Card>
        <Card title="Severity distribution">
          <Async state={defects}>
            {(d) => (
              <ResponsiveContainer width="100%" height={240}>
                <BarChart data={d.by_severity}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--grid)" />
                  <XAxis dataKey="name" fontSize={11} />
                  <YAxis allowDecimals={false} fontSize={11} />
                  <Tooltip />
                  <Bar dataKey="count" fill={PALETTE[2]} radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            )}
          </Async>
        </Card>
        <Card title="Quality-index distribution">
          <Async state={quality}>
            {(qd) => (
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={qd.quality_index_histogram}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--grid)" />
                  <XAxis dataKey="bin" fontSize={11} />
                  <YAxis allowDecimals={false} fontSize={11} />
                  <Tooltip />
                  <Bar dataKey="count" fill={PALETTE[1]} radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            )}
          </Async>
        </Card>
        <Card title="Rejection rate by profile" actions={<Link to="/products">Products →</Link>}>
          <Async state={defects}>
            {(d) => (
              <Table rows={d.by_profile} rowKey={(r) => r.profile_id} columns={[
                { label: "Profile", render: (r) => <Link to={`/inspections?profile_id=${r.profile_id}`}>{r.profile_id}</Link> },
                { label: "Total", key: "total" },
                { label: "PASS", key: "PASS" },
                { label: "FAIL", key: "FAIL" },
                { label: "Review", render: (r) => r["REVIEW REQUIRED"] + r.INCONCLUSIVE },
                { label: "Rejection", render: (r) => pct(r.rejection_rate) },
              ]} />
            )}
          </Async>
        </Card>
        <Card title="Predictions by model" actions={<Link to="/models">Model registry →</Link>}>
          <Async state={models}>
            {(m) => (
              <Table rows={m} rowKey={(r) => r.model} empty="No classifier predictions." columns={[
                { label: "Model", render: (r) => <span className="mono small">{r.model}</span> },
                { label: "Class counts", render: (r) => Object.entries(r).filter(([k]) => k !== "model").map(([k, v]) => `${k}: ${v}`).join(" · ") },
              ]} />
            )}
          </Async>
        </Card>
        <Card title="Drift monitoring" actions={<><Link to="/alerts" className="small">Alerts →</Link> <button className="btn btn-ghost btn-sm" onClick={runDrift}>Run check</button></>}>
          <Async state={drift}>
            {(rows) => (
              <Table rows={rows} empty="No drift checks yet." columns={[
                { label: "Time", render: (c) => new Date(c.created_at).toLocaleString() },
                { label: "Profile", key: "profile_id" },
                { label: "Input", render: (c) => (c.data_drift ? "⚠ drift" : "ok") },
                { label: "Prediction", render: (c) => (c.prediction_drift ? "⚠ drift" : "ok") },
                { label: "Interpretation", render: (c) => <span className="small">{c.details.interpretation}</span> },
              ]} />
            )}
          </Async>
        </Card>
      </div>
    </>
  );
}
