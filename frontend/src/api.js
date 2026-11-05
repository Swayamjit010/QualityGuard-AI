// Thin fetch wrapper around the QualityGuard REST API (JWT bearer auth).
const TOKEN_KEY = "qg_token";

export const getToken = () => localStorage.getItem(TOKEN_KEY);
export const setToken = (t) => (t ? localStorage.setItem(TOKEN_KEY, t) : localStorage.removeItem(TOKEN_KEY));

export class ApiError extends Error {
  constructor(status, message) {
    super(message);
    this.status = status;
  }
}

let onUnauthorized = () => {};
export const setUnauthorizedHandler = (fn) => (onUnauthorized = fn);

export async function api(path, { method = "GET", body, form, query, raw } = {}) {
  const url = new URL(path, window.location.origin);
  Object.entries(query || {}).forEach(([k, v]) => {
    if (v !== undefined && v !== null && v !== "") url.searchParams.set(k, v);
  });
  const headers = {};
  const t = getToken();
  if (t) headers.Authorization = `Bearer ${t}`;
  let payload;
  if (form) payload = form;
  else if (body !== undefined) {
    headers["Content-Type"] = "application/json";
    payload = JSON.stringify(body);
  }
  const res = await fetch(url, { method, headers, body: payload });
  if (res.status === 401 && path !== "/api/auth/login") onUnauthorized();
  if (!res.ok) {
    let msg = res.statusText;
    try {
      const j = await res.json();
      msg = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail);
    } catch {
      /* not JSON */
    }
    throw new ApiError(res.status, msg || `HTTP ${res.status}`);
  }
  if (raw) return res;
  const ct = res.headers.get("content-type") || "";
  return ct.includes("application/json") ? res.json() : res.text();
}

// URL usable in <img>/<a>/<video> (token as query param; the API accepts it for downloads)
export const authUrl = (path) => {
  if (!path) return "";
  const sep = path.includes("?") ? "&" : "?";
  return `${path}${sep}token=${encodeURIComponent(getToken() || "")}`;
};

export async function download(path, filename) {
  const res = await api(path, { raw: true });
  const blob = await res.blob();
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = filename || path.split("/").pop();
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 2000);
}

export function openEvents(onEvent) {
  const proto = window.location.protocol === "https:" ? "wss" : "ws";
  let ws;
  let closed = false;
  let retry;
  const connect = () => {
    ws = new WebSocket(`${proto}://${window.location.host}/api/ws/events?token=${encodeURIComponent(getToken() || "")}`);
    ws.onmessage = (m) => {
      try {
        const ev = JSON.parse(m.data);
        if (ev.type !== "ping") onEvent(ev);
      } catch {
        /* ignore */
      }
    };
    ws.onclose = (e) => {
      if (!closed && e.code !== 4401 && e.code !== 4403) retry = setTimeout(connect, 3000);
    };
  };
  connect();
  return () => {
    closed = true;
    clearTimeout(retry);
    ws && ws.close();
  };
}
