import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api";
import { Async, Badge, Card, ErrorBox, Field, PageHeader, Table, can, dt, useApi, useAuth, useToast } from "../components";

export default function Profiles() {
  const list = useApi("/api/profiles");
  const [sel, setSel] = useState("");
  useEffect(() => {
    if (!sel && list.data?.length) setSel(list.data[0].id);
  }, [list.data, sel]);
  return (
    <>
      <PageHeader title="Product profiles" subtitle="Per-product models, acceptance rules and severity rules. Every change creates a revision." />
      <div className="tabs">
        {(list.data || []).map((p) => (
          <button key={p.id} className={`tab ${sel === p.id ? "active" : ""}`} onClick={() => setSel(p.id)}>{p.name}</button>
        ))}
      </div>
      {sel && <ProfileEditor key={sel} id={sel} onSaved={list.reload} />}
    </>
  );
}

function ProfileEditor({ id, onSaved }) {
  const state = useApi(`/api/profiles/${id}`);
  const revs = useApi(`/api/profiles/${id}/revisions`);
  const { user } = useAuth();
  const toast = useToast();
  const [text, setText] = useState("");
  const [reason, setReason] = useState("");
  const [error, setError] = useState(null);
  useEffect(() => {
    if (state.data) setText(JSON.stringify(state.data.config, null, 2));
  }, [state.data]);

  const save = async () => {
    setError(null);
    let cfg;
    try {
      cfg = JSON.parse(text);
    } catch (e) {
      return setError(new Error(`Invalid JSON: ${e.message}`));
    }
    try {
      await api(`/api/profiles/${id}`, { method: "PUT", body: { config: cfg, reason: reason || "edited via UI" } });
      toast("Profile saved (new revision)", "success");
      setReason("");
      state.reload();
      revs.reload();
      onSaved();
    } catch (e) {
      setError(e);
    }
  };
  const rollback = async (v) => {
    try {
      await api("/api/models/rollback", { method: "POST", body: { profile_id: id, to_version: v } });
      toast(`Models rolled back to revision v${v}`, "success");
      state.reload();
      revs.reload();
    } catch (e) {
      toast(e.message, "error");
    }
  };
  const editable = can(user, "manage_profiles");

  return (
    <Async state={state}>
      {(p) => (
        <div className="grid grid-3">
          <div className="stack">
            <Card title="Summary" actions={<Link className="btn btn-primary btn-sm" to={`/inspect?profile=${p.id}`}>Inspect an item</Link>}>
              <div className="kv">
                <div><span className="muted">ID</span><b>{p.id}</b></div>
                <div><span className="muted">Version</span><b>v{p.version}</b></div>
                <div><span className="muted">Dataset</span><b>{p.dataset}</b></div>
                <div><span className="muted">Type</span><b>{p.product_type}</b></div>
                <div><span className="muted">Classifiers</span><b className="mono small">{(p.models.classifiers || []).join(", ") || "—"}</b></div>
                <div><span className="muted">Anomaly</span><b className="mono small">{p.models.anomaly || "—"}</b></div>
                <div><span className="muted">Fail at severity</span><b>{p.config.acceptance_rules?.fail_at_severity}</b></div>
                <div><span className="muted">Updated</span><b>{dt(p.updated_at)}</b></div>
              </div>
              <p className="small">{p.description}</p>
              {p.config.configuration_note && <p className="warn-text small">{p.config.configuration_note}</p>}
              {p.assumes_defect_present && <p className="muted small">{p.config.assumption_note}</p>}
              <div className="row gap-s wrap">{p.defect_classes.map((c) => <Badge key={c}>{c}</Badge>)}</div>
              <div className="row gap-s wrap small">
                <Link to={`/inspections?profile_id=${p.id}`}>Inspection history</Link>
                <Link to="/models">Models</Link>
                {p.models.anomaly && <Link to="/anomaly">Anomaly center</Link>}
                <Link to="/products">All products</Link>
              </div>
            </Card>
            <Card title="Revisions">
              <Async state={revs}>
                {(rows) => (
                  <Table rows={rows} rowKey={(r) => r.version} columns={[
                    { label: "v", key: "version" },
                    { label: "Reason", key: "reason" },
                    { label: "When", render: (r) => dt(r.created_at) },
                    { label: "", render: (r) => can(user, "approve_models") && r.version < p.version && JSON.stringify(r.models) !== JSON.stringify(p.models) && (
                      <button className="btn btn-ghost btn-sm" onClick={() => rollback(r.version)}>Roll back models</button>
                    ) },
                  ]} />
                )}
              </Async>
            </Card>
          </div>
          <Card title="Configuration (JSON)" className="span-2" actions={editable && <button className="btn btn-primary" onClick={save}>Save revision</button>}>
            <textarea className="code-editor" spellCheck={false} value={text} readOnly={!editable} onChange={(e) => setText(e.target.value)} />
            {editable && <Field label="Change reason"><input value={reason} maxLength={255} onChange={(e) => setReason(e.target.value)} placeholder="e.g. tightened severity bands per QA-123" /></Field>}
            <ErrorBox error={error} />
            <p className="muted small">Changing <code>models</code> requires the approve_models permission — prefer deploying from the Models page.</p>
          </Card>
        </div>
      )}
    </Async>
  );
}
