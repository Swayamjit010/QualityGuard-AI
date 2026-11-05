import { useEffect, useState } from "react";
import { api } from "../api";
import { Async, Badge, Card, InspectionLink, PageHeader, Table, can, dt, useApi, useAuth, useToast } from "../components";

const LEVEL = { critical: "fail", warning: "review", info: "info" };

export default function Alerts() {
  const [status, setStatus] = useState("active");
  const [level, setLevel] = useState("");
  const { user, refreshCounts } = useAuth();
  const toast = useToast();
  const state = useApi("/api/alerts", { query: { status, level, limit: 300 } });
  const { reload } = state;
  useEffect(() => {
    const h = (e) => e.detail.type === "alert" && reload();
    window.addEventListener("qg-event", h);
    return () => window.removeEventListener("qg-event", h);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  const act = async (a, action) => {
    try {
      await api(`/api/alerts/${a.id}/${action}`, { method: "POST" });
      reload();
      refreshCounts();
    } catch (e) {
      toast(e.message, "error");
    }
  };
  return (
    <>
      <PageHeader title="Alerts" subtitle="Critical defects, repeated defects, unfamiliar inputs, drift and failures.">
        <select value={status} onChange={(e) => setStatus(e.target.value)}>
          <option value="active">Active</option>
          <option value="open">Open</option>
          <option value="acknowledged">Acknowledged</option>
          <option value="resolved">Resolved</option>
          <option value="">All</option>
        </select>
        <select value={level} onChange={(e) => setLevel(e.target.value)}>
          <option value="">Any level</option>
          <option value="critical">Critical</option>
          <option value="warning">Warning</option>
          <option value="info">Info</option>
        </select>
      </PageHeader>
      <Card>
        <Async state={state}>
          {(rows) => (
            <Table rows={rows} empty="No alerts." columns={[
              { label: "Time", render: (a) => dt(a.created_at) },
              { label: "Level", render: (a) => <Badge kind={LEVEL[a.level]}>{a.level}</Badge> },
              { label: "Type", render: (a) => a.kind.replaceAll("_", " ") },
              { label: "Alert", render: (a) => <><b>{a.title}</b><div className="muted small">{a.message}</div></> },
              { label: "Inspection", render: (a) => <InspectionLink id={a.inspection_id} /> },
              { label: "Status", render: (a) => <Badge kind={a.status === "resolved" ? "pass" : a.status === "open" ? "review" : "info"}>{a.status}</Badge> },
              { label: "", render: (a) => can(user, "manage_alerts") && a.status !== "resolved" && (
                <div className="row gap-s">
                  {a.status === "open" && <button className="btn btn-ghost btn-sm" onClick={() => act(a, "acknowledge")}>Acknowledge</button>}
                  <button className="btn btn-ghost btn-sm" onClick={() => act(a, "resolve")}>Resolve</button>
                </div>
              ) },
            ]} />
          )}
        </Async>
      </Card>
    </>
  );
}
