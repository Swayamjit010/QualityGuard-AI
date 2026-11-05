import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { Bar, BarChart, CartesianGrid, Legend, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api, authUrl } from "../api";
import {
  Async, Badge, Card, Empty, InspectionLink, PageHeader, STATUS_COLORS, Stat, StatusBadge, Table, can, dt, num, pct,
  useApi, useAuth, useToast,
} from "../components";

export default function AnomalyCenter() {
  const profiles = useApi("/api/profiles");
  const anomalyProfiles = (profiles.data || []).filter((p) => p.models?.anomaly);
  const [pid, setPid] = useState("");
  const [days, setDays] = useState(0);
  useEffect(() => {
    if (!pid && anomalyProfiles.length) setPid(anomalyProfiles[0].id);
  }, [anomalyProfiles, pid]);
  const ov = useApi(pid ? "/api/analytics/anomaly-overview" : null, { query: { profile_id: pid, days: days || undefined } });
  const prof = anomalyProfiles.find((p) => p.id === pid);

  return (
    <>
      <PageHeader title="Anomaly detection center"
                  subtitle="Unsupervised engines trained only on defect-free images: thresholds, references, score distribution and unfamiliar inputs.">
        <select value={pid} onChange={(e) => setPid(e.target.value)} aria-label="Profile">
          {anomalyProfiles.map((p) => <option key={p.id} value={p.id}>{p.name} [{p.id}]</option>)}
        </select>
        <select value={days} onChange={(e) => setDays(+e.target.value)} aria-label="Period">
          <option value={0}>All time</option>
          {[1, 7, 30, 90].map((d) => <option key={d} value={d}>Last {d} days</option>)}
        </select>
      </PageHeader>
      {profiles.data && anomalyProfiles.length === 0 && <Empty>No product profile has an anomaly model configured.</Empty>}
      {pid && (
        <Async state={ov}>
          {(o) => (
            <div className="stack">
              <div className="stats">
                <Stat label="Inspections scored" value={o.inspections} />
                <Stat label="At or above threshold" value={o.above_threshold} hint={o.inspections ? pct(o.above_threshold / o.inspections) : ""} kind="fail" />
                <Stat label="Unfamiliar inputs" value={o.unfamiliar} kind={o.unfamiliar ? "review" : undefined} />
                <Stat label="Model" value={o.model?.backbone || "—"} hint={o.model?.run_id} />
              </div>
              {o.model ? <Thresholds m={o.model} /> : <Empty>The configured anomaly model has no training record.</Empty>}
              <Card title="Anomaly score distribution (score ÷ deployed threshold of the primary engine)">
                {o.inspections === 0 ? <Empty>No scored inspections for this profile yet.</Empty> : (
                  <ResponsiveContainer width="100%" height={260}>
                    <BarChart data={o.score_ratio_histogram.map((b) => ({ ...b, bin: b.to == null ? `≥${b.from}` : b.from.toFixed(1) }))}>
                      <CartesianGrid strokeDasharray="3 3" vertical={false} />
                      <XAxis dataKey="bin" tick={{ fontSize: 10 }} interval={2} />
                      <YAxis allowDecimals={false} width={32} />
                      <Tooltip />
                      <Legend />
                      <ReferenceLine x="1.0" stroke="#dc2626" label={{ value: "threshold", fontSize: 11, position: "top" }} />
                      {["PASS", "REVIEW REQUIRED", "FAIL", "INCONCLUSIVE"].map((s) => <Bar key={s} dataKey={s} stackId="a" fill={STATUS_COLORS[s]} />)}
                    </BarChart>
                  </ResponsiveContainer>
                )}
              </Card>
              <div className="grid grid-2">
                <Card title="Most anomalous inspections">
                  <Table rows={o.most_anomalous} empty="None yet" columns={[
                    { label: "Inspection", render: (i) => <InspectionLink id={i.id} /> },
                    { label: "Score / thr.", render: (i) => `${num(i.anomaly_score_ratio)}×` },
                    { label: "Status", render: (i) => <StatusBadge status={i.status} /> },
                    { label: "Review", render: (i) => i.review_status },
                    { label: "", render: (i) => <Link to={`/xai?inspection=${i.id}`} className="small">explain →</Link> },
                  ]} />
                </Card>
                <UnfamiliarAlerts pid={pid} />
              </div>
              {prof && <References profile={prof} />}
            </div>
          )}
        </Async>
      )}
    </>
  );
}

function Thresholds({ m }) {
  const rows = Object.entries(m.engines).map(([k, e]) => ({ engine: k, ...e }));
  return (
    <div className="grid grid-2">
      <Card title="Deployed thresholds (chosen on the validation split)">
        <Table rows={rows} rowKey={(r) => r.engine} columns={[
          { label: "Engine", render: (r) => <>{r.engine}{r.engine === m.primary_engine && <Badge kind="info">primary</Badge>}</> },
          { label: "Image threshold", render: (r) => num(r.deployed_threshold, 3) },
          { label: "Good items (val) mean / max", render: (r) => `${num(r.good_val_score_mean, 3)} / ${num(r.good_val_score_max, 3)}` },
          { label: "Pixel threshold", render: (r) => num(r.pixel_threshold, 3) },
        ]} />
        <p className="muted small">Rule: {rows[0]?.threshold_rule}. Review margin ±{pct(m.fusion?.review_margin, 0)} around each threshold. {m.fusion?.rule}</p>
        <p className="muted small">Thresholds are part of the trained model version. To change them, retrain or recalibrate and approve a new version on the <Link to="/models">Models</Link> page; acceptance rules live in <Link to="/profiles">Product profiles</Link>.</p>
      </Card>
      <Card title="Held-out test results (training record)">
        <Table rows={rows} rowKey={(r) => r.engine} columns={[
          { label: "Engine", render: (r) => r.engine },
          { label: "Image AUROC", render: (r) => num(r.test?.image_auroc, 3) },
          { label: "Recall", render: (r) => pct(r.test?.recall) },
          { label: "FPR", render: (r) => pct(r.test?.false_positive_rate) },
          { label: "Pixel AUROC", render: (r) => num(r.test_pixel?.pixel_auroc, 3) },
        ]} />
        {m.novelty && (
          <p className="small">Unfamiliar-input check: flagged {pct(m.novelty.foreign_detection_rate)} of {m.novelty.foreign_images} foreign images and {pct(m.novelty.test_in_distribution_flag_rate)} of in-distribution test images.</p>
        )}
        <p className="muted small">Split sizes: train {m.split_sizes?.train} (defect-free) · val {m.split_sizes?.val} · test {m.split_sizes?.test}. Trained {dt(m.completed_at)}.</p>
      </Card>
    </div>
  );
}

function UnfamiliarAlerts({ pid }) {
  const { user, refreshCounts } = useAuth();
  const toast = useToast();
  const st = useApi("/api/alerts", { query: { kind: "unfamiliar_pattern", status: "active", limit: 50 } });
  const rows = (st.data || []).filter((a) => !a.profile_id || a.profile_id === pid);
  const ack = async (a) => {
    try {
      await api(`/api/alerts/${a.id}/acknowledge`, { method: "POST" });
      st.reload();
      refreshCounts();
    } catch (e) {
      toast(e.message, "error");
    }
  };
  return (
    <Card title="Unfamiliar-pattern alerts" actions={<Link to="/reviews" className="btn btn-ghost btn-sm">Review queue →</Link>}>
      <Async state={st}>
        {() => (
          <Table rows={rows} empty="No active unfamiliar-pattern alerts" columns={[
            { label: "When", render: (a) => dt(a.created_at) },
            { label: "Alert", render: (a) => a.title },
            { label: "Inspection", render: (a) => a.inspection_id ? <InspectionLink id={a.inspection_id} /> : "—" },
            { label: "", render: (a) => can(user, "manage_alerts") && a.status === "open" ? <button className="btn btn-ghost btn-sm" onClick={() => ack(a)}>Acknowledge</button> : a.status },
          ]} />
        )}
      </Async>
    </Card>
  );
}

function References({ profile }) {
  const [split, setSplit] = useState("train");
  const ds = profile.dataset;
  const good = useApi(`/api/datasets/${ds}/samples`, { query: { split, label: "good", page_size: 12 } });
  const bad = useApi(`/api/datasets/${ds}/samples`, { query: { split: "test", page_size: 200 } });
  const anomalous = (bad.data?.items || []).filter((s) => s.label !== "good").slice(0, 12);
  return (
    <div className="grid grid-2">
      <Card title="Defect-free reference images" actions={
        <select value={split} onChange={(e) => setSplit(e.target.value)} aria-label="Split">
          <option value="train">training references</option><option value="test">test (good)</option>
        </select>}>
        <Async state={good}>
          {(g) => <Gallery items={g.items} />}
        </Async>
        <p className="muted small">{good.data?.total ?? "…"} images in this split. The anomaly engines model only these normal appearances.</p>
      </Card>
      <Card title="Known anomalous examples (test split, ground truth)">
        <Async state={bad}>{() => <Gallery items={anomalous} />}</Async>
      </Card>
    </div>
  );
}

function Gallery({ items }) {
  if (!items.length) return <Empty>No images</Empty>;
  return (
    <div className="gallery gallery-sm">
      {items.map((s) => (
        <figure key={s.url}>
          <img src={authUrl(s.url)} alt={`${s.label} ${s.file}`} loading="lazy" />
          <figcaption>{s.label}</figcaption>
        </figure>
      ))}
    </div>
  );
}
