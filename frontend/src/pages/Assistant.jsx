import { useEffect, useRef, useState } from "react";
import { api } from "../api";
import { Badge, Card, PageHeader, useApi } from "../components";
import { Markdown } from "../markdown";

const STARTERS = [
  "Summarise quality this week",
  "How many items failed today?",
  "Why was the last bottle rejected?",
  "Which defect is most common?",
  "What needs review?",
  "Any critical alerts?",
  "How accurate is the steel model?",
  "How do I check if an item is defective?",
];
const STORE = "qg_assistant_chat";

export default function Assistant() {
  const status = useApi("/api/assistant/status");
  const [messages, setMessages] = useState(() => {
    try {
      return JSON.parse(sessionStorage.getItem(STORE)) || [];
    } catch {
      return [];
    }
  });
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const end = useRef(null);
  const box = useRef(null);
  useEffect(() => {
    sessionStorage.setItem(STORE, JSON.stringify(messages.slice(-40)));
    end.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages, busy]);

  const ask = async (q) => {
    const question = (q ?? input).trim();
    if (!question || busy) return;
    setInput("");
    const history = messages.filter((m) => !m.error).slice(-10).map(({ role, content }) => ({ role, content }));
    setMessages((m) => [...m, { role: "user", content: question }]);
    setBusy(true);
    try {
      const r = await api("/api/assistant/chat", { method: "POST", body: { question, history } });
      setMessages((m) => [...m, { role: "assistant", content: r.answer, tools: r.tool_calls, mode: r.mode,
                                  suggestions: r.suggestions, note: r.note }]);
    } catch (e) {
      setMessages((m) => [...m, { role: "assistant", content: `Sorry, the request failed: ${e.message}`, error: true }]);
    } finally {
      setBusy(false);
      box.current?.focus();
    }
  };

  const last = messages[messages.length - 1];
  const followUps = last?.role === "assistant" && !busy ? (last.suggestions || []) : [];

  return (
    <>
      <PageHeader title="Quality assistant"
                  subtitle="Ask about inspections, defects, quality trends, models — or how to use the app. Every number comes from a database query.">
        {status.data && <Badge kind={status.data.llm_available ? "pass" : "info"}>{status.data.llm_available ? `Claude (${status.data.model})` : "offline engine"}</Badge>}
        {messages.length > 0 && <button className="btn btn-ghost" onClick={() => setMessages([])}>Clear chat</button>}
      </PageHeader>
      <Card className="chat-card">
        <div className="chat" aria-live="polite">
          {messages.length === 0 && (
            <div className="chat-empty">
              <p className="muted">Hi! I answer from your inspection records and model results. I don't look at images myself —
                to check an item, use <b>Inspect</b>. Try one of these:</p>
              <div className="suggestions">
                {STARTERS.map((s) => <button key={s} className="chip" onClick={() => ask(s)}>{s}</button>)}
              </div>
            </div>
          )}
          {messages.map((m, i) => (
            <div key={i} className={`msg msg-${m.role} ${m.error ? "msg-error" : ""}`}>
              <div className="msg-body">{m.role === "assistant" && !m.error ? <Markdown text={m.content} /> : m.content}</div>
              {m.note && <div className="muted small">{m.note}</div>}
              {m.tools?.length > 0 && (
                <details className="msg-tools">
                  <summary>{m.tools.length} data quer{m.tools.length > 1 ? "ies" : "y"} used</summary>
                  <ul>{m.tools.map((t, j) => <li key={j}><code>{t.tool}</code> {t.input && Object.keys(t.input).length ? JSON.stringify(t.input) : ""}</li>)}</ul>
                </details>
              )}
            </div>
          ))}
          {busy && <div className="msg msg-assistant"><div className="msg-body typing">Looking that up…</div></div>}
          {followUps.length > 0 && (
            <div className="suggestions">
              {followUps.map((s) => <button key={s} className="chip" onClick={() => ask(s)}>{s}</button>)}
            </div>
          )}
          <div ref={end} />
        </div>
        <form className="chat-input" onSubmit={(e) => { e.preventDefault(); ask(); }}>
          <input ref={box} value={input} maxLength={2000} onChange={(e) => setInput(e.target.value)}
                 placeholder="Ask a question, e.g. “why was the last bottle rejected?”" aria-label="Question" autoFocus />
          <button className="btn btn-primary" disabled={busy || !input.trim()}>Send</button>
        </form>
      </Card>
    </>
  );
}
