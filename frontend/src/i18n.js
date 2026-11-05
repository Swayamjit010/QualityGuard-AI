// Minimal localisation layer. English is the only complete language; to add one, copy the `en` block,
// translate the values, and register it in MESSAGES. Missing keys fall back to English, then to the key.
// The chosen language is stored per browser (localStorage "qg_lang") and sets <html lang>.

const en = {
  "nav.operations": "Operations",
  "nav.insight": "Insight",
  "nav.engineering": "Engineering",
  "nav.dashboard": "Dashboard",
  "nav.inspect": "Inspect",
  "nav.products": "Products",
  "nav.inspections": "Inspections",
  "nav.reviews": "Review queue",
  "nav.alerts": "Alerts",
  "nav.simulation": "Line simulation",
  "nav.analytics": "Analytics",
  "nav.anomaly": "Anomaly center",
  "nav.xai": "Explainable AI lab",
  "nav.reports": "Reports",
  "nav.assistant": "Assistant",
  "nav.models": "Models & training",
  "nav.datasets": "Datasets",
  "nav.profiles": "Product profiles",
  "nav.system": "System health",
  "nav.admin": "Administration",
  "shell.tagline": "Visual inspection",
  "shell.light": "☀ Light",
  "shell.dark": "☾ Dark",
  "shell.signout": "Sign out",
  "shell.gradio": "Gradio UI",
  "shell.notfound": "Page not found",
  "shell.language": "Language",
};

export const MESSAGES = { en };
export const LANGUAGES = [["en", "English"]];

let current = localStorage.getItem("qg_lang") || "en";
if (!MESSAGES[current]) current = "en";
document.documentElement.lang = current;

export const getLang = () => current;
export function setLang(lang) {
  if (!MESSAGES[lang]) return;
  current = lang;
  localStorage.setItem("qg_lang", lang);
  document.documentElement.lang = lang;
}

export function t(key, vars) {
  let s = MESSAGES[current]?.[key] ?? en[key] ?? key;
  if (vars) Object.entries(vars).forEach(([k, v]) => { s = s.replaceAll(`{${k}}`, String(v)); });
  return s;
}
