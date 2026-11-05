import { useState } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import { download } from "../api";
import {
  Async, Badge, Card, InspectionLink, PageHeader, Pager, ProfileSelect, SeverityBadge, StatusBadge, VerdictBadge, Table, can, dt, num,
  pct, useApi, useAuth, useToast,
} from "../components";

const FILTERS = ["verdict", "q", "profile_id", "status", "defect_class", "severity", "source", "review_status", "batch_id", "date_from", "date_to"];

export default function Inspections() {
  const [params, setParams] = useSearchParams();
  const [page, setPage] = useState(1);
  const { user } = useAuth();
  const toast = useToast();
  const navigate = useNavigate();
  const query = Object.fromEntries(FILTERS.map((k) => [k, params.get(k) || ""]));
  const state = useApi("/api/inspections", { query: { ...query, page, page_size: 25 } });
  const set = (k, v) => {
    const p = new URLSearchParams(params);
    v ? p.set(k, v) : p.delete(k);
    setParams(p);
    setPage(1);
  };
  const exportData = async (fmt) => {
    try {
      await download(`/api/inspections/export?fmt=${fmt}${query.profile_id ? `&profile_id=${query.profile_id}` : ""}${query.status ? `&status=${encodeURIComponent(query.status)}` : ""}${query.batch_id ? `&batch_id=${query.batch_id}` : ""}`, `inspections.${fmt}`);
    } catch (e) {
      toast(e.message, "error");
    }
  };

  return (
    <>
      <PageHeader title="Inspections" subtitle="Searchable, traceable history of every inspection.">
        {can(user, "export") && (
          <>
            <button className="btn btn-ghost" onClick={() => exportData("csv")}>Export CSV</button>
            <button className="btn btn-ghost" onClick={() => exportData("json")}>Export JSON</button>
          </>
        )}
      </PageHeader>
      <Card>
        <div className="filters">
          <input placeholder="Search ID, file, serial, defect…" value={query.q} onChange={(e) => set("q", e.target.value)} />
          <ProfileSelect value={query.profile_id} onChange={(v) => set("profile_id", v)} allowAll />
          {query.defect_class && <button className="chip" onClick={() => set("defect_class", "")} title="Remove filter">defect: {query.defect_class} ✕</button>}
          {query.batch_id && <button className="chip" onClick={() => set("batch_id", "")} title="Remove filter">batch: {query.batch_id.slice(0, 8)} ✕</button>}
          <select value={query.verdict || ""} onChange={(e) => set("verdict", e.target.value)} aria-label="Result">
            <option value="">Any result</option>
            {["NORMAL", "DEFECTIVE", "MANUAL INSPECTION REQUIRED"].map((s) => <option key={s}>{s}</option>)}
          </select>
          <select value={query.status} onChange={(e) => set("status", e.target.value)}>
            <option value="">Any disposition</option>
            {["PASS", "FAIL", "REVIEW REQUIRED", "INCONCLUSIVE"].map((s) => <option key={s}>{s}</option>)}
          </select>
          <select value={query.severity} onChange={(e) => set("severity", e.target.value)}>
            <option value="">Any severity</option>
            {["none", "low", "medium", "high", "critical"].map((s) => <option key={s}>{s}</option>)}
          </select>
          <select value={query.source} onChange={(e) => set("source", e.target.value)}>
            <option value="">Any source</option>
            {["single", "camera", "batch", "folder", "video", "api", "simulation"].map((s) => <option key={s}>{s}</option>)}
          </select>
          <select value={query.review_status} onChange={(e) => set("review_status", e.target.value)}>
            <option value="">Any review</option>
            {["pending", "reviewed", "not_required"].map((s) => <option key={s}>{s}</option>)}
          </select>
          <input type="date" value={query.date_from} onChange={(e) => set("date_from", e.target.value)} title="From" />
          <input type="date" value={query.date_to} onChange={(e) => set("date_to", e.target.value)} title="To" />
          {query.batch_id && <Badge kind="info">batch {query.batch_id.slice(0, 8)} <button className="link" onClick={() => set("batch_id", "")}>✕</button></Badge>}
        </div>
        <Async state={state}>
          {(d) => (
            <>
              <Table rows={d.items} empty="No inspections match these filters." onRowClick={(r) => navigate(`/inspections/${r.id}`)} columns={[
                { label: "Time", render: (r) => dt(r.created_at) },
                { label: "ID", render: (r) => <InspectionLink id={r.id} /> },
                { label: "Profile", key: "profile_id" },
                { label: "File", render: (r) => <span className="truncate" title={r.filename}>{r.filename}</span> },
                { label: "Result", render: (r) => <VerdictBadge verdict={r.verdict} /> },
                { label: "Disposition", render: (r) => <StatusBadge status={r.status} /> },
                { label: "Defect", render: (r) => r.defect_class || (r.defect_present ? "anomaly" : "—") },
                { label: "Severity", render: (r) => <SeverityBadge level={r.severity} /> },
                { label: "Conf.", render: (r) => pct(r.confidence) },
                { label: "QI", render: (r) => num(r.quality_index, 0) },
                { label: "Review", render: (r) => r.review_status === "pending" ? <Badge kind="review">pending</Badge> : r.human_decision ? <Badge kind={r.human_decision === "accept" ? "pass" : "fail"}>{r.human_decision}</Badge> : "—" },
                { label: "", render: (r) => <>{r.is_simulated && <Badge>sim</Badge>} <Badge>{r.source}</Badge></> },
              ]} />
              <Pager page={page} pageSize={25} total={d.total} onPage={setPage} />
            </>
          )}
        </Async>
      </Card>
    </>
  );
}
