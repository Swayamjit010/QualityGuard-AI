// Minimal, safe Markdown renderer for assistant answers. Builds React elements (no innerHTML), so text can
// never inject markup. Supports paragraphs, "- " bullets, "1. " numbered lists, **bold**, *italic* / _italic_,
// `code`, and [links](target). Relative links ("/inspections/..") navigate inside the app; "/api/.." links
// open in a new tab; anything else is rendered as plain text.
import { Link } from "react-router-dom";

const INLINE = /(\*\*[^*]+\*\*|`[^`]+`|\[[^\]]+\]\([^)\s]+\)|\*[^*\s][^*]*\*|_[^_\s][^_]*_)/g;

function inline(text, keyBase) {
  const out = [];
  let last = 0;
  let i = 0;
  for (const m of text.matchAll(INLINE)) {
    if (m.index > last) out.push(text.slice(last, m.index));
    const t = m[0];
    const k = `${keyBase}-${i++}`;
    if (t.startsWith("**")) out.push(<strong key={k}>{inline(t.slice(2, -2), k)}</strong>);
    else if (t.startsWith("`")) out.push(<code key={k}>{t.slice(1, -1)}</code>);
    else if (t.startsWith("[")) {
      const mm = t.match(/^\[([^\]]+)\]\(([^)\s]+)\)$/);
      const [label, href] = [mm[1], mm[2]];
      if (href.startsWith("/api/")) out.push(<a key={k} href={href} target="_blank" rel="noreferrer">{label}</a>);
      else if (href.startsWith("/")) out.push(<Link key={k} to={href}>{label}</Link>);
      else out.push(label);
    } else out.push(<em key={k}>{t.slice(1, -1)}</em>);
    last = m.index + t.length;
  }
  if (last < text.length) out.push(text.slice(last));
  return out;
}

export function Markdown({ text }) {
  const blocks = [];
  let list = null;
  const flush = () => {
    if (list) blocks.push(list.ordered ? <ol key={blocks.length}>{list.items}</ol> : <ul key={blocks.length}>{list.items}</ul>);
    list = null;
  };
  (text || "").split("\n").forEach((raw, n) => {
    const line = raw.trimEnd();
    const bullet = line.match(/^\s*[-•]\s+(.*)$/);
    const numbered = line.match(/^\s*\d+[.)]\s+(.*)$/);
    if (bullet || numbered) {
      const ordered = !!numbered;
      if (!list || list.ordered !== ordered) {
        flush();
        list = { ordered, items: [] };
      }
      list.items.push(<li key={n}>{inline((bullet || numbered)[1], `l${n}`)}</li>);
    } else if (!line.trim()) {
      flush();
    } else {
      flush();
      blocks.push(<p key={`p${n}`}>{inline(line, `p${n}`)}</p>);
    }
  });
  flush();
  return <div className="md">{blocks}</div>;
}
