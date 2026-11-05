import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api";
import {
  Async, Badge, Card, ErrorBox, Field, Modal, PageHeader, Pager, Table, bytes, can, dt, useApi, useAuth, useToast,
} from "../components";

export default function Admin() {
  const { user, refreshCounts } = useAuth();
  const tabs = [
    ["access", "Access requests", "manage_users"],
    ["users", "Users", "manage_users"],
    ["keys", "API keys", "manage_api_keys"],
    ["settings", "Settings", "manage_settings"],
    ["audit", "Audit trail", "view_logs"],
    ["logs", "System logs", "view_logs"],
    ["system", "System", "view_logs"],
    ["account", "My account", null],
  ].filter(([, , p]) => !p || can(user, p));
  const [tab, setTab] = useState(tabs[0][0]);
  return (
    <>
      <PageHeader title="Administration" />
      <div className="tabs">
        {tabs.map(([k, l]) => <button key={k} className={`tab ${tab === k ? "active" : ""}`} onClick={() => setTab(k)}>{l}</button>)}
      </div>
      {tab === "access" && <AccessRequests onChange={refreshCounts} />}
      {tab === "users" && <Users />}
      {tab === "keys" && <ApiKeys />}
      {tab === "settings" && <Settings />}
      {tab === "audit" && <Audit />}
      {tab === "logs" && <Logs />}
      {tab === "system" && <System />}
      {tab === "account" && <Account />}
    </>
  );
}

function AccessRequests({ onChange }) {
  const [status, setStatus] = useState("pending");
  const state = useApi("/api/access-requests", { query: { status: status || undefined } });
  const toast = useToast();
  const [notes, setNotes] = useState({});
  const [busy, setBusy] = useState(null);
  useEffect(() => {
    const h = (e) => e.detail?.type === "access_request" && state.reload();
    window.addEventListener("qg-event", h);
    return () => window.removeEventListener("qg-event", h);
  }, [state]);
  const decide = async (r, action) => {
    setBusy(r.id);
    try {
      await api(`/api/access-requests/${r.id}/decision`, { method: "POST", body: { action, note: notes[r.id] || "" } });
      toast(`${action === "approve" ? "Approved" : "Denied"}: ${r.user.username || r.user.email}`, action === "approve" ? "success" : "warn");
      state.reload();
      onChange && onChange();
    } catch (e) {
      toast(e.message, "error");
    } finally {
      setBusy(null);
    }
  };
  return (
    <Card title="Inspection access requests" actions={
      <select value={status} onChange={(e) => setStatus(e.target.value)} aria-label="Status filter">
        <option value="pending">Pending</option>
        <option value="approved">Approved</option>
        <option value="denied">Denied</option>
        <option value="">All</option>
      </select>}>
      <p className="muted small">
        Users who created their own account start as <b>Viewer</b> (they can look at results but not inspect).
        Approving a request gives them the <b>Inspector</b> role: view, run inspections, submit reviews, reports and alerts.
        Every decision is recorded in the audit trail.
      </p>
      <Async state={state}>
        {(d) => (
          <Table rows={d.items} rowKey={(r) => r.id} empty={status === "pending" ? "No pending requests." : "No requests."} columns={[
            { label: "Requested", render: (r) => dt(r.created_at) },
            { label: "User", render: (r) => <span><b>{r.user.username || "—"}</b><br /><span className="muted small">{r.user.email}</span></span> },
            { label: "Current role", render: (r) => <Badge kind="info">{r.user.role_title}</Badge> },
            { label: "Reason", render: (r) => r.reason || <span className="muted">—</span> },
            { label: "Status", render: (r) => <Badge kind={{ pending: "review", approved: "pass", denied: "fail" }[r.status]}>{r.status}</Badge> },
            { label: "Decision", render: (r) => r.status === "pending" ? (
              <div className="row gap-s wrap">
                <input placeholder="Note (optional)" value={notes[r.id] || ""} maxLength={1000} aria-label={`Note for ${r.user.email}`}
                       onChange={(e) => setNotes({ ...notes, [r.id]: e.target.value })} />
                <button className="btn btn-primary btn-sm" disabled={busy === r.id} onClick={() => decide(r, "approve")}>Approve</button>
                <button className="btn btn-danger btn-sm" disabled={busy === r.id} onClick={() => decide(r, "deny")}>Deny</button>
              </div>
            ) : (
              <span className="small">{r.decided_by} · {dt(r.decided_at)}{r.decision_note ? <><br />“{r.decision_note}”</> : null}</span>
            ) },
          ]} />
        )}
      </Async>
    </Card>
  );
}

function Users() {
  const users = useApi("/api/users");
  const roles = useApi("/api/roles");
  const [edit, setEdit] = useState(null);
  return (
    <Card title="Users" actions={<button className="btn btn-primary" onClick={() => setEdit({ email: "", full_name: "", role: "inspector", is_active: true, password: "" })}>Add user</button>}>
      <Async state={users}>
        {(rows) => (
          <Table rows={rows} onRowClick={(u) => setEdit({ ...u, password: "" })} columns={[
            { label: "Email", key: "email" },
            { label: "Username", render: (u) => u.username || "—" },
            { label: "Name", key: "full_name" },
            { label: "Role", render: (u) => <Badge kind="info">{u.role_title}</Badge> },
            { label: "Active", render: (u) => (u.is_active ? "yes" : <Badge kind="fail">disabled</Badge>) },
            { label: "Last login", render: (u) => dt(u.last_login_at) },
          ]} />
        )}
      </Async>
      <Async state={roles}>
        {(rs) => (
          <details className="small"><summary>Role permissions</summary>
            <Table rows={rs} rowKey={(r) => r.name} columns={[{ label: "Role", key: "title" }, { label: "Permissions", render: (r) => r.permissions.join(", ") }]} />
          </details>
        )}
      </Async>
      {edit && <UserModal u={edit} roles={roles.data || []} onClose={() => setEdit(null)} onDone={() => { setEdit(null); users.reload(); }} />}
    </Card>
  );
}

function UserModal({ u, roles, onClose, onDone }) {
  const [f, setF] = useState(u);
  const [error, setError] = useState(null);
  const set = (k) => (e) => setF({ ...f, [k]: e.target.type === "checkbox" ? e.target.checked : e.target.value });
  const save = async () => {
    try {
      const body = { email: f.email, full_name: f.full_name, role: f.role, is_active: f.is_active, password: f.password || null };
      await api(u.id ? `/api/users/${u.id}` : "/api/users", { method: u.id ? "PUT" : "POST", body });
      onDone();
    } catch (e) {
      setError(e);
    }
  };
  return (
    <Modal title={u.id ? `Edit ${u.email}` : "New user"} onClose={onClose}>
      <Field label="Email"><input type="email" value={f.email} disabled={!!u.id} onChange={set("email")} /></Field>
      <Field label="Full name"><input value={f.full_name} onChange={set("full_name")} /></Field>
      <Field label="Role">
        <select value={f.role} onChange={set("role")}>{roles.map((r) => <option key={r.name} value={r.name}>{r.title}</option>)}</select>
      </Field>
      <Field label={u.id ? "Reset password (optional)" : "Password"} hint="≥ 10 chars, upper, lower and digit">
        <input type="password" value={f.password} onChange={set("password")} autoComplete="new-password" />
      </Field>
      <label className="check"><input type="checkbox" checked={f.is_active} onChange={set("is_active")} /> Active</label>
      <ErrorBox error={error} />
      <button className="btn btn-primary" onClick={save}>Save</button>
    </Modal>
  );
}

function ApiKeys() {
  const keys = useApi("/api/api-keys");
  const [name, setName] = useState("");
  const [created, setCreated] = useState(null);
  const toast = useToast();
  const create = async () => {
    try {
      setCreated(await api("/api/api-keys", { method: "POST", body: { name: name || "integration" } }));
      setName("");
      keys.reload();
    } catch (e) {
      toast(e.message, "error");
    }
  };
  const revoke = async (k) => {
    await api(`/api/api-keys/${k.id}`, { method: "DELETE" }).catch((e) => toast(e.message, "error"));
    keys.reload();
  };
  return (
    <Card title="API keys" actions={<div className="row gap-s"><input placeholder="Key name" value={name} onChange={(e) => setName(e.target.value)} /><button className="btn btn-primary" onClick={create}>Create key</button></div>}>
      {created && (
        <div className="info-box">
          <b>New key (shown once):</b> <code className="select-all">{created.key}</code>
          <div className="small">{created.note} Example: <code>curl -H "X-API-Key: …" -F file=@part.png -F profile_id=bottle {window.location.origin}/api/inspections</code></div>
        </div>
      )}
      <Async state={keys}>
        {(rows) => (
          <Table rows={rows} empty="No API keys." columns={[
            { label: "Name", key: "name" },
            { label: "Prefix", render: (k) => <span className="mono">{k.prefix}…</span> },
            { label: "Owner", key: "user" },
            { label: "Created", render: (k) => dt(k.created_at) },
            { label: "Last used", render: (k) => dt(k.last_used_at) },
            { label: "", render: (k) => (k.revoked ? <Badge kind="fail">revoked</Badge> : <button className="btn btn-ghost btn-sm" onClick={() => revoke(k)}>Revoke</button>) },
          ]} />
        )}
      </Async>
    </Card>
  );
}

function Settings() {
  const state = useApi("/api/settings");
  const toast = useToast();
  const [drafts, setDrafts] = useState({});
  const save = async (key) => {
    try {
      await api(`/api/settings/${key}`, { method: "PUT", body: JSON.parse(drafts[key]) });
      toast(`${key} saved`, "success");
      state.reload();
    } catch (e) {
      toast(e.message, "error");
    }
  };
  return (
    <Async state={state}>
      {(s) => (
        <div className="grid grid-2">
          {Object.entries(s).map(([k, v]) => (
            <Card key={k} title={k.replaceAll("_", " ")} actions={<button className="btn btn-primary btn-sm" onClick={() => save(k)}>Save</button>}>
              <textarea className="code-editor code-sm" spellCheck={false} value={drafts[k] ?? JSON.stringify(v, null, 2)}
                        onChange={(e) => setDrafts({ ...drafts, [k]: e.target.value })} />
            </Card>
          ))}
        </div>
      )}
    </Async>
  );
}

function Audit() {
  const [action, setAction] = useState("");
  const [page, setPage] = useState(1);
  const state = useApi("/api/system/audit", { query: { action, page, page_size: 50 } });
  return (
    <Card title="Audit trail" actions={<input placeholder="Filter action prefix (e.g. model.)" value={action} onChange={(e) => { setAction(e.target.value); setPage(1); }} />}>
      <Async state={state}>
        {(d) => (
          <>
            <Table rows={d.items} columns={[
              { label: "Time", render: (e) => dt(e.created_at) },
              { label: "User", key: "user_email" },
              { label: "Action", render: (e) => <code>{e.action}</code> },
              { label: "Entity", render: (e) => <EntityLink type={e.entity_type} id={String(e.entity_id)} /> },
              { label: "Details", render: (e) => <span className="small mono truncate" title={JSON.stringify(e.details)}>{JSON.stringify(e.details)}</span> },
              { label: "IP", key: "ip" },
            ]} />
            <Pager page={page} pageSize={50} total={d.total} onPage={setPage} />
          </>
        )}
      </Async>
    </Card>
  );
}

function Logs() {
  const [level, setLevel] = useState("");
  const [q, setQ] = useState("");
  const state = useApi("/api/system/logs", { query: { level, q, limit: 500 } });
  return (
    <Card title="Application logs" actions={
      <div className="row gap-s">
        <select value={level} onChange={(e) => setLevel(e.target.value)}>
          <option value="">All levels</option>{["INFO", "WARNING", "ERROR"].map((l) => <option key={l}>{l}</option>)}
        </select>
        <input placeholder="Search" value={q} onChange={(e) => setQ(e.target.value)} />
        <button className="btn btn-ghost btn-sm" onClick={state.reload}>Refresh</button>
      </div>
    }>
      <Async state={state}>
        {(rows) => <pre className="log">{rows.map((r) => `${r.time.slice(0, 19)} ${r.level.padEnd(7)} ${r.logger}: ${r.message}`).join("\n") || "(empty)"}</pre>}
      </Async>
    </Card>
  );
}

function System() {
  const state = useApi("/api/system/info");
  return (
    <Async state={state}>
      {(s) => (
        <div className="grid grid-2">
          <Card title="Runtime">
            <div className="kv">
              <div><span className="muted">Python</span><b>{s.python}</b></div>
              <div><span className="muted">PyTorch</span><b>{s.torch}</b></div>
              <div><span className="muted">Device</span><b>{s.device?.device || JSON.stringify(s.device)}</b></div>
              <div><span className="muted">Platform</span><b className="small">{s.platform}</b></div>
              <div><span className="muted">Uptime</span><b>{Math.round(s.uptime_s / 60)} min</b></div>
              <div><span className="muted">Inspections</span><b>{s.inspections}</b></div>
              <div><span className="muted">Assistant LLM</span><b>{s.assistant_llm ? "available" : "offline mode"}</b></div>
            </div>
          </Card>
          <Card title="Storage">
            <div className="kv">
              <div><span className="muted">Files</span><b>{s.storage.files}</b></div>
              <div><span className="muted">Used</span><b>{bytes(s.storage.bytes)}</b></div>
              <div><span className="muted">Disk free</span><b>{bytes(s.storage.disk_free_bytes)} / {bytes(s.storage.disk_total_bytes)}</b></div>
            </div>
          </Card>
          <Card title="Loaded models (in memory)" className="span-2">
            <Table rows={s.loaded_models} rowKey={(m, i) => i} empty="No models loaded yet (loaded lazily on first inspection)." columns={
              Object.keys(s.loaded_models[0] || { run_id: 1 }).map((k) => ({ label: k, render: (m) => String(m[k]) }))
            } />
          </Card>
        </div>
      )}
    </Async>
  );
}

function Account() {
  const { user } = useAuth();
  const [cur, setCur] = useState("");
  const [pw, setPw] = useState("");
  const [error, setError] = useState(null);
  const toast = useToast();
  const change = async () => {
    setError(null);
    try {
      await api("/api/auth/me/password", { method: "POST", body: { current_password: cur, new_password: pw } });
      toast("Password changed", "success");
      setCur("");
      setPw("");
    } catch (e) {
      setError(e);
    }
  };
  return (
    <div className="grid grid-2">
      <Card title="Profile">
        <div className="kv">
          <div><span className="muted">Email</span><b>{user.email}</b></div>
          <div><span className="muted">Name</span><b>{user.full_name}</b></div>
          <div><span className="muted">Role</span><b>{user.role_title}</b></div>
        </div>
        <div className="row gap-s wrap small">{user.permissions.map((p) => <Badge key={p}>{p}</Badge>)}</div>
      </Card>
      <Card title="Change password">
        <Field label="Current password"><input type="password" value={cur} onChange={(e) => setCur(e.target.value)} autoComplete="current-password" /></Field>
        <Field label="New password" hint="≥ 10 chars, upper, lower and digit"><input type="password" value={pw} onChange={(e) => setPw(e.target.value)} autoComplete="new-password" /></Field>
        <ErrorBox error={error} />
        <button className="btn btn-primary" disabled={!cur || !pw} onClick={change}>Change password</button>
      </Card>
    </div>
  );
}

// Links audit entries to the record they concern, where the app has a page for it.
function EntityLink({ type, id }) {
  const label = `${type} ${id.slice(0, 24)}`;
  const to = type === "inspection" ? `/inspections/${id}`
    : type === "profile" ? "/profiles"
      : type === "product" ? `/products/${id.replace(/^custom_/, "")}`
        : type === "model_version" || type === "training_job" ? "/models"
          : type === "batch" ? `/inspections?batch_id=${id}`
            : type === "dataset" ? "/datasets" : type === "report" ? "/reports" : type === "alert" ? "/alerts" : null;
  return to && id ? <Link to={to}>{label}</Link> : label;
}
