import { useCallback, useEffect, useState } from "react";
import { NavLink, Navigate, Route, Routes, useLocation, useNavigate } from "react-router-dom";
import { t } from "./i18n";
import { api, getToken, openEvents, setToken, setUnauthorizedHandler } from "./api";
import { AuthContext, Loading, can, useToast } from "./components";
import Admin from "./pages/Admin";
import Alerts from "./pages/Alerts";
import AnomalyCenter from "./pages/AnomalyCenter";
import Analytics from "./pages/Analytics";
import Assistant from "./pages/Assistant";
import Dashboard from "./pages/Dashboard";
import Datasets from "./pages/Datasets";
import ExplainLab from "./pages/ExplainLab";
import AccessBanner from "./pages/AccessRequest";
import Inspect from "./pages/Inspect";
import InspectAI from "./pages/InspectAI";
import InspectionDetail from "./pages/InspectionDetail";
import Inspections from "./pages/Inspections";
import Landing, { Logo } from "./pages/Landing";
import Login from "./pages/Login";
import Models from "./pages/Models";
import Products from "./pages/Products";
import Profiles from "./pages/Profiles";
import Reports from "./pages/Reports";
import Reviews from "./pages/Reviews";
import Simulation from "./pages/Simulation";
import SystemHealth from "./pages/SystemHealth";

const NAV = [
  { section: "nav.operations" },
  { to: "/", label: "nav.dashboard", icon: "◧", perm: "view", end: true },
  { to: "/inspect", label: "nav.inspect", icon: "⊕", perm: "inspect" },
  { to: "/products", label: "nav.products", icon: "▣", perm: "view" },
  { to: "/inspections", label: "nav.inspections", icon: "☰", perm: "view" },
  { to: "/reviews", label: "nav.reviews", icon: "✓", perm: "view", badge: "reviews" },
  { to: "/alerts", label: "nav.alerts", icon: "⚠", perm: "view", badge: "alerts" },
  { to: "/simulation", label: "nav.simulation", icon: "▶", perm: "inspect" },
  { section: "nav.insight" },
  { to: "/analytics", label: "nav.analytics", icon: "◔", perm: "view" },
  { to: "/anomaly", label: "nav.anomaly", icon: "◎", perm: "view" },
  { to: "/xai", label: "nav.xai", icon: "◐", perm: "view" },
  { to: "/reports", label: "nav.reports", icon: "▤", perm: "reports" },
  { to: "/assistant", label: "nav.assistant", icon: "✦", perm: "view" },
  { section: "nav.engineering" },
  { to: "/models", label: "nav.models", icon: "◈", perm: "view" },
  { to: "/datasets", label: "nav.datasets", icon: "▦", perm: "view" },
  { to: "/profiles", label: "nav.profiles", icon: "⚙", perm: "view" },
  { to: "/system", label: "nav.system", icon: "♥", perm: "view_logs" },
  { to: "/admin", label: "nav.admin", icon: "⛭", badge: "access", anyPerm: ["manage_users", "view_logs", "manage_settings", "manage_api_keys"] },
];

export default function App() {
  const [user, setUser] = useState(null);
  const [booting, setBooting] = useState(!!getToken());
  const navigate = useNavigate();
  const location = useLocation();
  const toast = useToast();
  const [counts, setCounts] = useState({ alerts: 0, reviews: 0, access: 0 });
  const [theme, setTheme] = useState(localStorage.getItem("qg_theme") || "light");

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    localStorage.setItem("qg_theme", theme);
  }, [theme]);

  const logout = useCallback(async () => {
    try {
      await api("/api/auth/logout", { method: "POST" });
    } catch {
      /* already invalid */
    }
    setToken(null);
    setUser(null);
    navigate("/login");
  }, [navigate]);

  useEffect(() => {
    setUnauthorizedHandler(() => {
      setToken(null);
      setUser(null);
    });
    if (!getToken()) return;
    api("/api/auth/me")
      .then(setUser)
      .catch(() => setToken(null))
      .finally(() => setBooting(false));
  }, []);

  const refreshCounts = useCallback(() => {
    if (!user) return;
    Promise.all([
      api("/api/alerts", { query: { status: "open", limit: 500 } }),
      api("/api/reviews/queue", { query: { page_size: 1 } }),
      can(user, "manage_users") ? api("/api/access-requests", { query: { status: "pending" } }) : Promise.resolve({ pending: 0 }),
    ])
      .then(([a, r, ar]) => setCounts({ alerts: a.length, reviews: r.total, access: ar.pending }))
      .catch(() => {});
  }, [user]);

  useEffect(() => {
    if (!user) return;
    refreshCounts();
    return openEvents((ev) => {
      if (ev.type === "alert") {
        toast(`${ev.data.level === "critical" ? "🚨 " : ""}${ev.data.title}`, ev.data.level === "critical" ? "error" : "warn");
        refreshCounts();
      } else if (ev.type === "review" || (ev.type === "inspection" && ev.data.manual_review_required)) {
        refreshCounts();
      } else if (ev.type === "access_request") {
        if (ev.data.user_id === user.id && ev.data.status !== "pending") {
          // my request was decided: reload my permissions so the Inspect pages appear (or the banner updates)
          api("/api/auth/me").then(setUser).catch(() => {});
          toast(ev.data.status === "approved" ? "Inspection access approved — you can now run inspections."
            : "Your inspection-access request was denied.", ev.data.status === "approved" ? "success" : "warn");
        } else if (can(user, "manage_users") && ev.data.status === "pending") {
          toast("New inspection-access request — see Administration → Access requests.", "info");
        }
        refreshCounts();
      }
      window.dispatchEvent(new CustomEvent("qg-event", { detail: ev }));
    });
  }, [user, refreshCounts, toast]);

  if (booting) return <Loading />;
  if (!user) {
    return (
      <Routes>
        <Route path="/" element={<Landing />} />
        <Route path="*" element={<Login onLogin={(u) => { setUser(u); navigate(location.state?.from || "/"); }} />} />
      </Routes>
    );
  }

  const visible = NAV.filter((n) => n.section || (n.perm ? can(user, n.perm) : n.anyPerm.some((p) => can(user, p))));
  return (
    <AuthContext.Provider value={{ user, setUser, logout, refreshCounts }}>
      <div className="shell">
        <aside className="sidebar">
          <div className="brand">
            <Logo size={28} />
            <div>
              <div className="brand-name">QualityGuard AI</div>
              <div className="brand-sub">{t("shell.tagline")}</div>
            </div>
          </div>
          <nav>
            {visible.map((n, i) =>
              n.section ? (
                <div key={i} className="nav-section">{t(n.section)}</div>
              ) : (
                <NavLink key={n.to} to={n.to} end={n.end} className="nav-link">
                  <span className="nav-icon">{n.icon}</span>
                  <span>{t(n.label)}</span>
                  {n.badge && counts[n.badge] > 0 && <span className="nav-badge">{counts[n.badge]}</span>}
                </NavLink>
              )
            )}
          </nav>
          <div className="sidebar-foot">
            <div className="user-chip">
              <div className="avatar">{(user.full_name || user.email)[0].toUpperCase()}</div>
              <div className="user-meta">
                <div className="user-name">{user.full_name || user.email}</div>
                <div className="muted small">{user.role_title}</div>
              </div>
            </div>
            <div className="row gap-s">
              <button className="btn btn-ghost btn-sm" onClick={() => setTheme(theme === "dark" ? "light" : "dark")}>
                {theme === "dark" ? t("shell.light") : t("shell.dark")}
              </button>
              <a className="btn btn-ghost btn-sm" href="/gradio/" target="_blank" rel="noreferrer">{t("shell.gradio")}</a>
              <button className="btn btn-ghost btn-sm" onClick={logout}>{t("shell.signout")}</button>
            </div>
          </div>
        </aside>
        <main className="content">
          {!can(user, "inspect") && <AccessBanner />}
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/inspect" element={<InspectAI />} />
            <Route path="/inspect/advanced" element={<Inspect />} />
            <Route path="/inspections" element={<Inspections />} />
            <Route path="/inspections/:id" element={<InspectionDetail />} />
            <Route path="/reviews" element={<Reviews />} />
            <Route path="/alerts" element={<Alerts />} />
            <Route path="/simulation" element={<Simulation />} />
            <Route path="/analytics" element={<Analytics />} />
            <Route path="/anomaly" element={<AnomalyCenter />} />
            <Route path="/xai" element={<ExplainLab />} />
            <Route path="/system" element={<SystemHealth />} />
            <Route path="/reports" element={<Reports />} />
            <Route path="/assistant" element={<Assistant />} />
            <Route path="/models/*" element={<Models />} />
            <Route path="/datasets/*" element={<Datasets />} />
            <Route path="/profiles" element={<Profiles />} />
            <Route path="/products/*" element={<Products />} />
            <Route path="/admin" element={<Admin />} />
            <Route path="/login" element={<Navigate to="/" />} />
            <Route path="*" element={<div className="empty">{t("shell.notfound")}. <NavLink to="/">Go to the dashboard</NavLink> or <NavLink to="/inspect">inspect an item</NavLink>.</div>} />
          </Routes>
        </main>
      </div>
    </AuthContext.Provider>
  );
}
