import { useState } from "react";
import { Link } from "react-router-dom";
import { authUrl } from "../api";
import { Async, Card, Empty, PageHeader, Pager, ProfileSelect, SeverityBadge, StatusBadge, dt, num, pct, short, useApi } from "../components";

export default function Reviews() {
  const [profile, setProfile] = useState("");
  const [page, setPage] = useState(1);
  const [order, setOrder] = useState("priority");
  const state = useApi("/api/reviews/queue", { query: { profile_id: profile, page, page_size: 24, order } });
  return (
    <>
      <PageHeader title="Review queue" subtitle="Inspections the AI could not decide on its own. Priority order puts the most informative items for retraining first (active learning).">
        <ProfileSelect value={profile} onChange={(v) => { setProfile(v); setPage(1); }} allowAll />
        <select value={order} onChange={(e) => { setOrder(e.target.value); setPage(1); }} aria-label="Order">
          <option value="priority">Highest learning value first</option>
          <option value="oldest">Oldest first</option>
          <option value="newest">Newest first</option>
        </select>
      </PageHeader>
      <Async state={state}>
        {(d) => d.items.length ? (
          <>
            <div className="review-grid">
              {d.items.map((r) => (
                <Link key={r.id} to={`/inspections/${r.id}`} className="review-card">
                  <img src={authUrl(`/api/inspections/${r.id}/images/original`)} alt="" loading="lazy" />
                  <div className="review-body">
                    <div className="row between"><StatusBadge status={r.status} /><span className="mono small">{short(r.id)}</span></div>
                    <div><b>{r.defect_class || (r.defect_present ? "anomaly" : "no defect")}</b> <SeverityBadge level={r.severity} /></div>
                    <div className="muted small">{r.profile_id} · conf {pct(r.confidence)} · ratio {num(r.anomaly_score_ratio)}</div>
                    <div className="muted small">{dt(r.created_at)}{r.unfamiliar ? " · unfamiliar" : ""}</div>
                    {r.priority_reasons?.length > 0 && <div className="small" title="Active-learning priority terms">priority {num(r.priority)} · {r.priority_reasons.join(", ")}</div>}
                  </div>
                </Link>
              ))}
            </div>
            <Pager page={page} pageSize={24} total={d.total} onPage={setPage} />
          </>
        ) : <Card><Empty>Nothing waiting for review. 🎉</Empty></Card>}
      </Async>
    </>
  );
}
