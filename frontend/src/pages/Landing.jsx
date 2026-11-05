import { Link } from "react-router-dom";

export const Logo = ({ size = 32, light }) => (
  <svg viewBox="0 0 32 32" width={size} height={size} aria-hidden="true">
    <path d="M16 2 4 7v8c0 7.5 5.1 13.4 12 15 6.9-1.6 12-7.5 12-15V7z" fill={light ? "#fff" : "#1d6fe0"} />
    <path d="m10 16 4 4 8-9" stroke={light ? "#1d6fe0" : "#fff"} strokeWidth="3" fill="none" strokeLinecap="round" strokeLinejoin="round" />
  </svg>
);

const CAPABILITIES = [
  ["Known-defect classification", "Transfer-learning CNN ensembles (MobileNetV2, EfficientNet, ResNet and more) identify which trained defect type is present, with calibrated confidence."],
  ["Unseen-defect detection", "PatchCore and PaDiM learn only from defect-free products, so they flag abnormalities no one labelled in advance and outline where they are."],
  ["Unfamiliar-input awareness", "Feature-space distance and energy checks recognise inputs unlike the training data and send them to a person instead of guessing."],
  ["Explainable decisions", "Grad-CAM, Grad-CAM++, Score-CAM and anomaly maps are generated from the real inference run for every decision."],
  ["Configurable quality rules", "Severity, PASS / FAIL / REVIEW and the quality index come from per-product rules the manufacturer controls — no hard-coded tolerances."],
  ["Human in the loop", "Uncertain or disputed items go to a review queue; verified corrections feed an approval-gated retraining workflow."],
];

const WORKFLOW = ["Capture", "Validate image", "Classify + detect anomalies", "Localise + measure", "Check reliability", "Apply product rules", "Record, alert, report"];

const MODES = ["Single image", "Batch upload", "Folder", "Video (sampled frames, annotated output)", "Live webcam / USB camera", "Production-line simulation", "Authenticated REST API", "Golden-reference comparison"];

export default function Landing() {
  return (
    <div className="landing">
      <header className="landing-nav">
        <div className="row gap-s" style={{ alignItems: "center" }}>
          <Logo size={30} />
          <b>QualityGuard AI</b>
        </div>
        <nav className="row gap-s">
          <a href="#capabilities" className="btn btn-ghost btn-sm">Capabilities</a>
          <a href="#workflow" className="btn btn-ghost btn-sm">Workflow</a>
          <Link to="/login" className="btn btn-ghost btn-sm">Register</Link>
          <Link to="/login" className="btn btn-primary btn-sm">Sign in</Link>
        </nav>
      </header>

      <section className="landing-hero">
        <div>
          <p className="landing-eyebrow">Industrial visual quality inspection</p>
          <h1>Find defects. Show why. Keep people in charge.</h1>
          <p className="landing-lead">
            QualityGuard AI combines supervised defect classification with unsupervised anomaly detection, explains every
            decision visually and routes uncertain items to an inspector — with full traceability from image to report.
          </p>
          <div className="row gap-s">
            <Link to="/login" className="btn btn-primary">Sign in to the platform</Link>
            <a href="/api/docs" className="btn btn-ghost">REST API reference</a>
          </div>
          <p className="muted small">An AI-assisted inspection tool. It is not certified for production release decisions without validation on the manufacturer's own line.</p>
        </div>
        <img className="landing-shot" src="/preview-dashboard.png" alt="Dashboard showing inspection totals, a quality trend chart, defect distribution and recent alerts" />
      </section>

      <section id="capabilities" className="landing-section">
        <h2>Core capabilities</h2>
        <div className="landing-grid">
          {CAPABILITIES.map(([t, d]) => (
            <div key={t} className="card landing-card"><h3>{t}</h3><p>{d}</p></div>
          ))}
        </div>
      </section>

      <section id="workflow" className="landing-section">
        <h2>Inspection workflow</h2>
        <ol className="landing-flow">
          {WORKFLOW.map((w, i) => <li key={w}><span>{i + 1}</span>{w}</li>)}
        </ol>
      </section>

      <section className="landing-section landing-two">
        <div>
          <h2>Inspection modes</h2>
          <ul className="landing-list">{MODES.map((m) => <li key={m}>{m}</li>)}</ul>
        </div>
        <div>
          <h2>Technology</h2>
          <ul className="landing-list">
            <li>PyTorch · torchvision · timm models with a configurable registry</li>
            <li>FastAPI backend, SQLAlchemy (SQLite or PostgreSQL), Alembic migrations</li>
            <li>React dashboard, Gradio quick-inspection UI, WebSocket live events</li>
            <li>Role-based access (5 roles), API keys, audit trail, PDF / CSV / JSON reports</li>
            <li>Runs on an ordinary CPU; a GPU only makes it faster</li>
          </ul>
        </div>
      </section>

      <footer className="landing-foot muted small">QualityGuard AI · KIIT School of Computer Engineering project</footer>
    </div>
  );
}
