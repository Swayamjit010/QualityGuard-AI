import { useEffect, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { Area, AreaChart, Bar, BarChart, CartesianGrid, Cell, Legend, Pie, PieChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import {
  Async, Badge, Card, Empty, InspectionLink, PALETTE, PageHeader, ProfileSelect, STATUS_COLORS, SeverityBadge, Stat,
  StatusBadge, VerdictBadge, Table, dt, num, pct, useApi,
} from "../components";

export default function Dashboard() {
  const [days, setDays] = useState(30);
  const [profile, setProfile] = useState("");
  const [sim, setSim] = useState(true);
  const state = useApi("/api/analytics/dashboard", { query: { days, profile_id: profile, include_simulated: sim } });
  const navigate = useNavigate();
  const { reload } = state;

  // live refresh (debounced) when inspections arrive
  useEffect(() => {
    let t;
    const h = (e) => {
      if (["inspection", "alert", "review"].includes(e.detail.type)) {
        clearTimeout(t);
        t = setTimeout(reload, 1500);
      }
    };
    window.addEventListener("qg-event", h);
    return () => {
      window.removeEventListener("qg-event", h);
      clearTimeout(t);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <>
      <PageHeader title="Quality dashboard" subtitle="Live overview of inspection outcomes, defects and alerts.">
        <Link className="btn btn-primary" to="/inspect">Check an item</Link>
        <ProfileSelect value={profile} onChange={setProfile} allowAll />
        <select value={days} onChange={(e) => setDays(+e.target.value)}>
          {[1, 7, 30, 90, 365].map((d) => <option key={d} value={d}>Last {d} day{d > 1 ? "s" : ""}</option>)}
        </select>
        <label className="check"><input type="checkbox" checked={sim} onChange={(e) => setSim(e.target.checked)} /> include simulated</label>
      </PageHeader>
      <Async state={state}>
        {(d) => (
          <>
            <div className="stats">
              <Stat label="Inspections" value={d.total} hint={d.simulated ? `${d.simulated} simulated` : "real inputs"} to={`/inspections${profile ? `?profile_id=${profile}` : ""}`} />
              <Stat label="Normal" value={d.verdicts.NORMAL} kind="pass" hint={`acceptance rate ${pct(d.acceptance_rate)}`} to={`/inspections?verdict=NORMAL${profile ? `&profile_id=${profile}` : ""}`} />
              <Stat label="Defective" value={d.verdicts.DEFECTIVE} kind="fail" hint={`${d.by_status.FAIL} rejected by profile rules`} to={`/inspections?verdict=DEFECTIVE${profile ? `&profile_id=${profile}` : ""}`} />
              <Stat label="Manual inspection" value={d.verdicts["MANUAL INSPECTION REQUIRED"]} kind="review" hint={`${d.pending_reviews} pending review`} to={`/inspections?verdict=${encodeURIComponent("MANUAL INSPECTION REQUIRED")}`} />
              <Stat label="Avg latency" value={`${num(d.avg_latency_ms, 0)} ms`} hint="per image, CPU" to="/system" />
            </div>
            {d.total === 0 ? (
              <Card>
                <Empty>
                  No inspections yet. <Link to="/inspect">Run an inspection</Link> or start a{" "}
                  <Link to="/simulation">production-line simulation</Link>.
                </Empty>
              </Card>
            ) : (
              <div className="grid grid-3">
                <Card title="Outcome mix">
                  <ResponsiveContainer width="100%" height={240}>
                    <PieChart>
                      <Pie data={Object.entries(d.by_status).filter(([, v]) => v).map(([name, value]) => ({ name, value }))}
                           dataKey="value" nameKey="name" innerRadius={55} outerRadius={90} paddingAngle={2}>
                        {Object.entries(d.by_status).filter(([, v]) => v).map(([k]) => <Cell key={k} fill={STATUS_COLORS[k]} />)}
                      </Pie>
                      <Tooltip />
                      <Legend />
                    </PieChart>
                  </ResponsiveContainer>
                </Card>
                <Card title="Daily trend" className="span-2">
                  <ResponsiveContainer width="100%" height={240}>
                    <AreaChart data={d.trend}>
                      <CartesianGrid strokeDasharray="3 3" stroke="var(--grid)" />
                      <XAxis dataKey="period" fontSize={11} />
                      <YAxis allowDecimals={false} fontSize={11} />
                      <Tooltip />
                      <Legend />
                      <Area type="monotone" dataKey="pass" stackId="1" name="PASS" stroke={STATUS_COLORS.PASS} fill={STATUS_COLORS.PASS} fillOpacity={0.5} />
                      <Area type="monotone" dataKey="fail" stackId="1" name="FAIL" stroke={STATUS_COLORS.FAIL} fill={STATUS_COLORS.FAIL} fillOpacity={0.5} />
                      <Area type="monotone" dataKey="review" stackId="1" name="REVIEW" stroke={STATUS_COLORS["REVIEW REQUIRED"]} fill={STATUS_COLORS["REVIEW REQUIRED"]} fillOpacity={0.5} />
                      <Area type="monotone" dataKey="inconclusive" stackId="1" name="INCONCLUSIVE" stroke={STATUS_COLORS.INCONCLUSIVE} fill={STATUS_COLORS.INCONCLUSIVE} fillOpacity={0.5} />
                    </AreaChart>
                  </ResponsiveContainer>
                </Card>
                <Card title="Defect distribution">
                  {d.defect_distribution.length ? (
                    <ResponsiveContainer width="100%" height={240}>
                      <BarChart data={d.defect_distribution} layout="vertical" margin={{ left: 30 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="var(--grid)" />
                        <XAxis type="number" allowDecimals={false} fontSize={11} />
                        <YAxis type="category" dataKey="name" width={110} fontSize={11} />
                        <Tooltip />
                        <Bar dataKey="count" fill={PALETTE[0]} radius={[0, 4, 4, 0]} />
                      </BarChart>
                    </ResponsiveContainer>
                  ) : <Empty>No defects recorded.</Empty>}
                </Card>
                <Card title="Insights" className="span-2">
                  <ul className="insights">
                    {d.insights.map((x, i) => <li key={i}>{x.text}</li>)}
                  </ul>
                </Card>
              </div>
            )}
            <div className="grid grid-3">
              <Card title="Recent inspections" className="span-2" actions={<Link to="/inspections">View all</Link>}>
                <Table rows={d.recent} empty="No inspections" onRowClick={(r) => navigate(`/inspections/${r.id}`)} columns={[
                  { label: "Time", render: (r) => dt(r.created_at) },
                  { label: "ID", render: (r) => <InspectionLink id={r.id} /> },
                  { label: "Profile", key: "profile_id" },
                  { label: "Result", render: (r) => <VerdictBadge verdict={r.verdict} /> },
                { label: "Disposition", render: (r) => <StatusBadge status={r.status} /> },
                  { label: "Defect", render: (r) => r.defect_class || "—" },
                  { label: "Severity", render: (r) => <SeverityBadge level={r.severity} /> },
                  { label: "", render: (r) => r.is_simulated && <Badge>sim</Badge> },
                ]} />
              </Card>
              <div className="stack">
                <Card title="Active alerts" actions={<Link to="/alerts">All alerts</Link>}>
                  {d.alerts.length ? (
                    <ul className="alert-list">
                      {d.alerts.map((a) => (
                        <li key={a.id} className={`alert-item lvl-${a.level}`}>
                          <div className="alert-title">{a.title}</div>
                          <div className="muted small">{dt(a.created_at)} · {a.status}</div>
                        </li>
                      ))}
                    </ul>
                  ) : <Empty>No active alerts.</Empty>}
                </Card>
                <Card title="Deployed models" actions={<Link to="/models">Manage</Link>}>
                  {d.deployed_models.length ? (
                    <ul className="plain">
                      {d.deployed_models.map((m) => (
                        <li key={m.run_id}><span className="mono small">{m.run_id}</span> <Badge kind="info">{m.task}</Badge></li>
                      ))}
                    </ul>
                  ) : <Empty>No deployed models.</Empty>}
                </Card>
              </div>
            </div>
          </>
        )}
      </Async>
    </>
  );
}
