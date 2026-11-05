import { createContext, useCallback, useContext, useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "./api";

// ------------------------------------------------------------------ auth context
export const AuthContext = createContext(null);
export const useAuth = () => useContext(AuthContext);
export const can = (user, perm) => !!user && user.permissions.includes(perm);

// ------------------------------------------------------------------ toasts
const ToastContext = createContext(() => {});
export const useToast = () => useContext(ToastContext);

export function ToastProvider({ children }) {
  const [items, setItems] = useState([]);
  const push = useCallback((text, kind = "info") => {
    const id = Math.random().toString(36).slice(2);
    setItems((x) => [...x, { id, text, kind }]);
    setTimeout(() => setItems((x) => x.filter((t) => t.id !== id)), 5000);
  }, []);
  return (
    <ToastContext.Provider value={push}>
      {children}
      <div className="toasts">
        {items.map((t) => (
          <div key={t.id} className={`toast toast-${t.kind}`}>{t.text}</div>
        ))}
      </div>
    </ToastContext.Provider>
  );
}

// ------------------------------------------------------------------ data hook
export function useApi(path, opts = {}, deps = []) {
  const [state, setState] = useState({ data: null, error: null, loading: !!path });
  const [tick, setTick] = useState(0);
  const key = JSON.stringify(opts);
  useEffect(() => {
    if (!path) return;
    let alive = true;
    setState((s) => ({ ...s, loading: true }));
    api(path, opts)
      .then((data) => alive && setState({ data, error: null, loading: false }))
      .catch((error) => alive && setState({ data: null, error, loading: false }));
    return () => {
      alive = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [path, key, tick, ...deps]);
  return { ...state, reload: () => setTick((t) => t + 1) };
}

export function useInterval(fn, ms) {
  const ref = useRef(fn);
  ref.current = fn;
  useEffect(() => {
    if (!ms) return;
    const id = setInterval(() => ref.current(), ms);
    return () => clearInterval(id);
  }, [ms]);
}

// ------------------------------------------------------------------ formatting
export const pct = (v, d = 1) => (v === null || v === undefined ? "—" : `${(v * 100).toFixed(d)}%`);
export const num = (v, d = 2) => (v === null || v === undefined ? "—" : Number(v).toFixed(d));
export const dt = (s) => (s ? new Date(s).toLocaleString() : "—");
export const short = (id) => (id ? id.slice(0, 8) : "—");
export const bytes = (n) => {
  if (n === null || n === undefined) return "—";
  const u = ["B", "KB", "MB", "GB", "TB"];
  let i = 0;
  while (n >= 1024 && i < u.length - 1) {
    n /= 1024;
    i++;
  }
  return `${n.toFixed(i ? 1 : 0)} ${u[i]}`;
};

// ------------------------------------------------------------------ UI atoms
export const STATUS_CLASS = { PASS: "pass", FAIL: "fail", "REVIEW REQUIRED": "review", INCONCLUSIVE: "muted" };

export function StatusBadge({ status }) {
  return <span className={`badge badge-${STATUS_CLASS[status] || "muted"}`}>{status || "—"}</span>;
}

export const VERDICT_CLASS = { NORMAL: "pass", DEFECTIVE: "fail", "MANUAL INSPECTION REQUIRED": "review" };

// Primary inspection output: NORMAL / DEFECTIVE / MANUAL INSPECTION REQUIRED
export function VerdictBadge({ verdict }) {
  return <span className={`badge badge-${VERDICT_CLASS[verdict] || "muted"}`}>{verdict || "—"}</span>;
}

export function SeverityBadge({ level }) {
  return <span className={`badge sev-${level || "none"}`}>{level || "none"}</span>;
}

export function Badge({ children, kind = "muted" }) {
  return <span className={`badge badge-${kind}`}>{children}</span>;
}

export function Card({ title, actions, children, className = "" }) {
  return (
    <section className={`card ${className}`}>
      {(title || actions) && (
        <header className="card-head">
          <h3>{title}</h3>
          <div className="card-actions">{actions}</div>
        </header>
      )}
      <div className="card-body">{children}</div>
    </section>
  );
}

export function Stat({ label, value, hint, kind, to }) {
  const body = (
    <>
      <div className="stat-label">{label}</div>
      <div className="stat-value">{value}</div>
      {hint && <div className="stat-hint">{hint}</div>}
    </>
  );
  return to
    ? <Link to={to} className={`stat stat-link ${kind ? `stat-${kind}` : ""}`} title={`Open: ${label}`}>{body}</Link>
    : <div className={`stat ${kind ? `stat-${kind}` : ""}`}>{body}</div>;
}

export function PageHeader({ title, subtitle, children }) {
  return (
    <div className="page-header">
      <div>
        <h1>{title}</h1>
        {subtitle && <p className="muted">{subtitle}</p>}
      </div>
      <div className="page-actions">{children}</div>
    </div>
  );
}

export const Loading = () => <div className="loading">Loading…</div>;

export function ErrorBox({ error }) {
  if (!error) return null;
  return <div className="error-box">{error.message || String(error)}</div>;
}

export function Empty({ children }) {
  return <div className="empty">{children}</div>;
}

export function Async({ state, children }) {
  if (state.error) return <ErrorBox error={state.error} />;
  if (state.loading && !state.data) return <Loading />;
  if (!state.data) return null;
  return children(state.data);
}

export function Table({ columns, rows, empty = "No data", rowKey = (r, i) => r.id ?? i, onRowClick }) {
  if (!rows || !rows.length) return <Empty>{empty}</Empty>;
  return (
    <div className="table-wrap">
      <table className="table">
        <thead>
          <tr>{columns.map((c) => <th key={c.key || c.label} style={{ width: c.width }}>{c.label}</th>)}</tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={rowKey(r, i)} onClick={onRowClick ? () => onRowClick(r) : undefined}
                className={onRowClick ? "clickable" : ""}>
              {columns.map((c) => (
                <td key={c.key || c.label}>{c.render ? c.render(r) : r[c.key]}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function Pager({ page, pageSize, total, onPage }) {
  const pages = Math.max(1, Math.ceil(total / pageSize));
  return (
    <div className="pager">
      <button className="btn btn-ghost" disabled={page <= 1} onClick={() => onPage(page - 1)}>‹ Prev</button>
      <span className="muted">Page {page} / {pages} · {total} items</span>
      <button className="btn btn-ghost" disabled={page >= pages} onClick={() => onPage(page + 1)}>Next ›</button>
    </div>
  );
}

export function Modal({ title, onClose, children, wide }) {
  useEffect(() => {
    const h = (e) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", h);
    return () => window.removeEventListener("keydown", h);
  }, [onClose]);
  return (
    <div className="modal-backdrop" onMouseDown={(e) => e.target === e.currentTarget && onClose()}>
      <div className={`modal ${wide ? "modal-wide" : ""}`} role="dialog" aria-modal="true">
        <header className="card-head">
          <h3>{title}</h3>
          <button className="btn btn-ghost" onClick={onClose} aria-label="Close">✕</button>
        </header>
        <div className="card-body">{children}</div>
      </div>
    </div>
  );
}

export function Field({ label, children, hint }) {
  return (
    <label className="field">
      <span className="field-label">{label}</span>
      {children}
      {hint && <span className="field-hint">{hint}</span>}
    </label>
  );
}

export function InspectionLink({ id }) {
  return id ? <Link to={`/inspections/${id}`} className="mono">{short(id)}</Link> : "—";
}

export function ProgressBar({ value }) {
  return (
    <div className="progress"><div className="progress-bar" style={{ width: `${Math.round((value || 0) * 100)}%` }} /></div>
  );
}

export function ProfileSelect({ value, onChange, allowAll, required }) {
  const { data } = useApi("/api/profiles");
  return (
    <select value={value || ""} onChange={(e) => onChange(e.target.value)} required={required}>
      {allowAll && <option value="">All profiles</option>}
      {!allowAll && !value && <option value="">Select a profile…</option>}
      {(data || []).map((p) => (
        <option key={p.id} value={p.id}>{p.name} [{p.id}]</option>
      ))}
    </select>
  );
}

export function JsonView({ value }) {
  return <pre className="json">{JSON.stringify(value, null, 2)}</pre>;
}

// colour palette for charts (categorical, colour-blind aware)
export const PALETTE = ["#1d6fe0", "#14b8a6", "#f59e0b", "#e11d48", "#8b5cf6", "#64748b", "#0ea5e9", "#84cc16"];
export const STATUS_COLORS = { PASS: "#16a34a", FAIL: "#dc2626", "REVIEW REQUIRED": "#d97706", INCONCLUSIVE: "#94a3b8" };
