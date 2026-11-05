// Banner for signed-in users who cannot run inspections yet: request the Inspector role and follow its status.
import { useCallback, useEffect, useState } from "react";
import { api } from "../api";
import { useAuth, useToast } from "../components";

export default function AccessBanner() {
  const { user } = useAuth();
  const toast = useToast();
  const [mine, setMine] = useState(null);
  const [reason, setReason] = useState("");
  const [open, setOpen] = useState(false);
  const [busy, setBusy] = useState(false);

  const load = useCallback(() => api("/api/access-requests/mine").then(setMine).catch(() => setMine([])), []);
  useEffect(() => { load(); }, [load]);
  useEffect(() => {
    const h = (e) => e.detail?.type === "access_request" && e.detail.data?.user_id === user.id && load();
    window.addEventListener("qg-event", h);
    return () => window.removeEventListener("qg-event", h);
  }, [load, user.id]);

  if (mine === null) return null;
  const last = mine[0];
  const send = async () => {
    setBusy(true);
    try {
      await api("/api/access-requests", { method: "POST", body: { reason } });
      toast("Request sent. An administrator will review it.", "success");
      setOpen(false);
      setReason("");
      load();
    } catch (e) {
      toast(e.message, "error");
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="access-banner" role="status" data-testid="access-banner">
      {last?.status === "pending" ? (
        <span>⏳ Your request for <b>inspection access</b> (sent {new Date(last.created_at).toLocaleString()}) is waiting for an administrator.</span>
      ) : (
        <>
          <span>
            {last?.status === "denied"
              ? <>Your last request for inspection access was <b>denied</b>{last.decision_note ? <> — “{last.decision_note}”</> : null}. You can send a new one.</>
              : <>You can view results, but you can't run inspections yet.</>}
          </span>
          {open ? (
            <span className="row gap-s">
              <input value={reason} maxLength={1000} onChange={(e) => setReason(e.target.value)}
                     placeholder="Why do you need access? (optional, e.g. line / shift)" aria-label="Reason" style={{ minWidth: 280 }} />
              <button className="btn btn-primary btn-sm" disabled={busy} onClick={send}>{busy ? "Sending…" : "Send request"}</button>
              <button className="btn btn-ghost btn-sm" onClick={() => setOpen(false)}>Cancel</button>
            </span>
          ) : (
            <button className="btn btn-primary btn-sm" onClick={() => setOpen(true)}>Request inspection access</button>
          )}
        </>
      )}
    </div>
  );
}
