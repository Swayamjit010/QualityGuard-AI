import { useEffect, useState } from "react";
import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api } from "../api";
import {
  Badge, Card, ErrorBox, Field, InspectionLink, PageHeader, ProfileSelect, ProgressBar, Stat, StatusBadge, Table, num,
  useApi,
} from "../components";

export default function Simulation() {
  const [profile, setProfile] = useState("bottle");
  const [interval, setIntervalS] = useState(1);
  const [count, setCount] = useState(40);
  const [rate, setRate] = useState("0.3");
  const [job, setJob] = useState(null);
  const [events, setEvents] = useState([]);
  const [error, setError] = useState(null);
  const running = useApi("/api/jobs", { query: { kind: "simulation", limit: 1 } });

  useEffect(() => {
    const j = running.data?.[0];
    if (j && ["queued", "running"].includes(j.status)) setJob({ job_id: j.id, batch_id: j.params.batch_id, status: j.status });
  }, [running.data]);

  useEffect(() => {
    const h = (e) => {
      const ev = e.detail;
      if (ev.type === "simulation") setEvents((x) => [ev.data, ...x].slice(0, 300));
      if (ev.type === "job" && job && ev.data.id === job.job_id && ev.data.status) setJob((j) => ({ ...j, status: ev.data.status, progress: ev.data.progress ?? j.progress }));
    };
    window.addEventListener("qg-event", h);
    return () => window.removeEventListener("qg-event", h);
  }, [job]);

  const start = async () => {
    setError(null);
    setEvents([]);
    try {
      const r = await api("/api/simulation/start", { method: "POST", body: {
        profile_id: profile, interval_s: +interval, count: +count, defect_rate: rate === "" ? null : +rate } });
      setJob({ ...r, status: "running", progress: 0 });
    } catch (e) {
      setError(e);
    }
  };
  const stop = () => api(`/api/jobs/${job.job_id}/stop`, { method: "POST" }).catch(setError);
  const last = events[0];
  const isRunning = job && ["queued", "running"].includes(job.status);
  const chart = [...events].reverse().map((e) => ({ seq: e.seq, throughput: +e.throughput_per_min.toFixed(1), latency: Math.round(e.latency_ms) }));

  return (
    <>
      <PageHeader title="Production-line simulation"
                  subtitle="Replays held-out dataset images through the real pipeline at a set arrival rate. All records are flagged SIMULATED." />
      <div className="grid grid-3">
        <Card title="Configuration">
          <Field label="Profile"><ProfileSelect value={profile} onChange={setProfile} /></Field>
          <div className="row gap">
            <Field label="Interval (s)"><input type="number" min="0" max="60" step="0.1" value={interval} onChange={(e) => setIntervalS(e.target.value)} /></Field>
            <Field label="Items"><input type="number" min="1" max="2000" value={count} onChange={(e) => setCount(e.target.value)} /></Field>
          </div>
          <Field label="Defect injection rate" hint="Share of items drawn from defective test images. Empty = natural test-set mix.">
            <input type="number" min="0" max="1" step="0.05" value={rate} onChange={(e) => setRate(e.target.value)} />
          </Field>
          {isRunning ? <button className="btn btn-danger btn-block" onClick={stop}>Stop line</button>
            : <button className="btn btn-primary btn-block" disabled={!profile} onClick={start}>Start line</button>}
          <ErrorBox error={error} />
          {job && <><div className="muted small">Job {job.job_id.slice(0, 8)} · {job.status}</div><ProgressBar value={job.progress ?? (last ? last.seq / count : 0)} /></>}
        </Card>
        <div className="span-2 stack">
          <div className="stats">
            <Stat label="Items" value={last?.seq ?? 0} />
            <Stat label="Accepted" value={last?.accepted ?? 0} kind="pass" />
            <Stat label="Rejected" value={last?.rejected ?? 0} kind="fail" />
            <Stat label="Held for review" value={(last?.review ?? 0) + (last?.inconclusive ?? 0)} kind="review" />
            <Stat label="Throughput" value={`${num(last?.throughput_per_min, 1)}/min`} />
          </div>
          <Card title="Line signal">
            <div className={`line-signal sig-${last?.signal || "idle"}`}>
              {last ? <>{last.signal} · {last.serial} · {last.status}{last.defect_class ? ` · ${last.defect_class}` : ""}</> : "Idle"}
            </div>
            {chart.length > 1 && (
              <ResponsiveContainer width="100%" height={180}>
                <LineChart data={chart}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--grid)" />
                  <XAxis dataKey="seq" fontSize={11} />
                  <YAxis yAxisId="l" fontSize={11} />
                  <YAxis yAxisId="r" orientation="right" fontSize={11} />
                  <Tooltip />
                  <Line yAxisId="l" type="monotone" dataKey="throughput" name="items/min" stroke="#1d6fe0" dot={false} />
                  <Line yAxisId="r" type="monotone" dataKey="latency" name="latency ms" stroke="#f59e0b" dot={false} />
                </LineChart>
              </ResponsiveContainer>
            )}
          </Card>
        </div>
      </div>
      <Card title="Event stream">
        <Table rows={events} rowKey={(e) => e.seq} empty="Start the line to see events." columns={[
          { label: "#", key: "seq" },
          { label: "Serial", render: (e) => <span className="mono small">{e.serial}</span> },
          { label: "Signal", render: (e) => <Badge kind={e.signal === "PASS" ? "pass" : e.signal === "REJECT" ? "fail" : "review"}>{e.signal}</Badge> },
          { label: "Status", render: (e) => <StatusBadge status={e.status} /> },
          { label: "Predicted", render: (e) => e.defect_class || "—" },
          { label: "Ground truth", render: (e) => e.ground_truth },
          { label: "Latency", render: (e) => `${num(e.latency_ms, 0)} ms` },
          { label: "Inspection", render: (e) => <InspectionLink id={e.inspection_id} /> },
        ]} />
      </Card>
    </>
  );
}
