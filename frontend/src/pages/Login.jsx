import { useState } from "react";
import { api, setToken } from "../api";

export default function Login({ onLogin }) {
  const [mode, setMode] = useState("login");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [username, setUsername] = useState("");
  const [error, setError] = useState("");
  const [info, setInfo] = useState("");
  const [busy, setBusy] = useState(false);

  const submit = async (e) => {
    e.preventDefault();
    setError("");
    setInfo("");
    setBusy(true);
    try {
      if (mode === "register") {
        await api("/api/auth/register", { method: "POST", body: { email, username, password } });
        setInfo("Account created. Sign in, then use “Request inspection access” — an administrator will approve or deny it.");
        setEmail(username);
        setPassword("");
        setMode("login");
      } else {
        const r = await api("/api/auth/login", { method: "POST", body: { email, password } });
        setToken(r.access_token);
        onLogin(r.user);
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="login-page">
      <div className="login-hero">
        <svg viewBox="0 0 32 32" width="56" height="56" aria-hidden="true">
          <path d="M16 2 4 7v8c0 7.5 5.1 13.4 12 15 6.9-1.6 12-7.5 12-15V7z" fill="#fff" fillOpacity=".95" />
          <path d="m10 16 4 4 8-9" stroke="#1d6fe0" strokeWidth="3" fill="none" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
        <h1>QualityGuard AI</h1>
        <p>Deep-learning industrial defect detection with explainable decisions, human review and traceable quality reports.</p>
        <ul>
          <li>Transfer-learning classifiers (MobileNetV2, EfficientNet, ResNet) + anomaly detection</li>
          <li>Grad-CAM and anomaly heat-maps for every decision</li>
          <li>Inspector review workflow, alerts, analytics and PDF reports</li>
        </ul>
      </div>
      <form className="login-card" onSubmit={submit}>
        <h2>{mode === "login" ? "Sign in" : "Create account"}</h2>
        {mode === "register" ? (
          <>
            <label className="field">
              <span className="field-label">Email</span>
              <input type="email" required value={email} onChange={(e) => setEmail(e.target.value)} autoComplete="email" autoFocus />
            </label>
            <label className="field">
              <span className="field-label">Username</span>
              <input required minLength={3} maxLength={32} pattern="[A-Za-z0-9][A-Za-z0-9_.\-]*" value={username}
                     onChange={(e) => setUsername(e.target.value)} autoComplete="username" />
              <span className="field-hint">3–32 characters: letters, digits, “.”, “_” or “-”.</span>
            </label>
          </>
        ) : (
          <label className="field">
            <span className="field-label">Email or username</span>
            <input required value={email} onChange={(e) => setEmail(e.target.value)} autoComplete="username" autoFocus />
          </label>
        )}
        <label className="field">
          <span className="field-label">Password</span>
          <input type="password" required value={password} onChange={(e) => setPassword(e.target.value)}
                 autoComplete={mode === "login" ? "current-password" : "new-password"} />
          {mode === "register" && <span className="field-hint">≥ 10 characters, upper- and lower-case and a digit.</span>}
        </label>
        {error && <div className="error-box">{error}</div>}
        {info && <div className="info-box">{info}</div>}
        <button className="btn btn-primary btn-block" disabled={busy}>{busy ? "Please wait…" : mode === "login" ? "Sign in" : "Register"}</button>
        <button type="button" className="btn btn-link" onClick={() => { setError(""); setMode(mode === "login" ? "register" : "login"); }}>
          {mode === "login" ? "No account? Create one" : "Have an account? Sign in"}
        </button>
        <p className="muted small">First start: initial accounts are written to <code>storage/INITIAL_CREDENTIALS.txt</code>.</p>
      </form>
    </div>
  );
}
