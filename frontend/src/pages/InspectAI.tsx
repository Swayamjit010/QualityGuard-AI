// AI Inspection — the primary page: upload → INSPECT → NORMAL / DEFECTIVE / MANUAL INSPECTION REQUIRED.
// Every value shown here comes from the stored inspection record produced by the trained models.
import { useEffect, useMemo, useRef, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import {
  api, authUrl, download, fix, pct,
  type InspectionDetail, type InspectionResult, type Profile, type Verdict,
} from "../lib/qg";

const ACCEPT = ".png,.jpg,.jpeg,.bmp,.tif,.tiff,.webp";

const VERDICT_STYLE: Record<Verdict, { ring: string; bg: string; icon: string; label: string }> = {
  NORMAL: { ring: "tw-border-verdict-normal", bg: "tw-bg-verdict-normal", icon: "✓", label: "NORMAL" },
  DEFECTIVE: { ring: "tw-border-verdict-defective", bg: "tw-bg-verdict-defective", icon: "✕", label: "DEFECTIVE" },
  "MANUAL INSPECTION REQUIRED": {
    ring: "tw-border-verdict-manual", bg: "tw-bg-verdict-manual", icon: "?", label: "MANUAL INSPECTION REQUIRED",
  },
};

function Panel({ title, children, className = "" }: { title?: string; children: React.ReactNode; className?: string }) {
  return (
    <section className={`tw-rounded-xl tw-border tw-border-solid tw-border-qg-line tw-bg-qg-card tw-p-4 tw-shadow-sm ${className}`}>
      {title && <h3 className="tw-m-0 tw-mb-3 tw-text-sm tw-font-semibold tw-uppercase tw-tracking-wide tw-text-qg-muted">{title}</h3>}
      {children}
    </section>
  );
}

function Stat({ label, value, hint }: { label: string; value: React.ReactNode; hint?: string }) {
  return (
    <div className="tw-rounded-lg tw-bg-qg-panel2 tw-p-3" title={hint}>
      <div className="tw-text-xs tw-text-qg-muted">{label}</div>
      <div className="tw-mt-1 tw-text-lg tw-font-semibold tw-text-qg-ink">{value}</div>
      {hint && <div className="tw-mt-1 tw-text-[11px] tw-leading-snug tw-text-qg-muted">{hint}</div>}
    </div>
  );
}

export default function InspectAI() {
  const [profiles, setProfiles] = useState<Profile[]>([]);
  const [params] = useSearchParams();
  const [profile, setProfile] = useState<string>(params.get("profile") || "auto");
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string>("");
  const [drag, setDrag] = useState(false);
  const [busy, setBusy] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<InspectionDetail | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    api<Profile[]>("/api/profiles").then(setProfiles).catch(() => setProfiles([]));
  }, []);

  useEffect(() => {
    if (!busy) return;
    const t0 = performance.now();
    const h = window.setInterval(() => setElapsed((performance.now() - t0) / 1000), 100);
    return () => window.clearInterval(h);
  }, [busy]);

  const pick = (f?: File | null) => {
    if (!f) return;
    setFile(f);
    setResult(null);
    setError(null);
    if (preview) URL.revokeObjectURL(preview);
    setPreview(URL.createObjectURL(f));
  };

  const run = async () => {
    if (!file) return;
    setBusy(true);
    setError(null);
    setResult(null);
    try {
      const fd = new FormData();
      fd.append("file", file);
      fd.append("profile_id", profile);
      fd.append("options", JSON.stringify({ cam_methods: ["gradcam", "gradcam_pp"] }));
      setResult(await api<InspectionDetail>("/api/inspect", { method: "POST", form: fd }));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="tw-mx-auto tw-max-w-[1400px]">
      <header className="tw-mb-5 tw-flex tw-flex-wrap tw-items-center tw-gap-4">
        <div className="tw-flex tw-items-center tw-gap-3">
          <div className="tw-flex tw-h-11 tw-w-11 tw-items-center tw-justify-center tw-rounded-xl tw-bg-qg-brand tw-text-xl tw-font-bold tw-text-white">QG</div>
          <div>
            <h1 className="tw-m-0 tw-text-2xl tw-font-bold tw-text-qg-ink">QualityGuard AI · Inspection</h1>
            <p className="tw-m-0 tw-text-sm tw-text-qg-muted">
              Upload an image of a supported component or surface, then press INSPECT.
            </p>
          </div>
        </div>
        <div className="tw-ml-auto tw-flex tw-items-center tw-gap-2 tw-text-sm">
          <Link to="/inspect/advanced" className="btn btn-ghost btn-sm">Batch · camera · video · reference →</Link>
        </div>
      </header>

      <div className="tw-grid tw-gap-5 lg:tw-grid-cols-[380px_1fr]">
        {/* ------------------------------------------------ input column */}
        <Panel title="1 · Image">
          <div
            role="button"
            tabIndex={0}
            aria-label="Choose an image to inspect"
            onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && inputRef.current?.click()}
            onClick={() => inputRef.current?.click()}
            onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
            onDragLeave={() => setDrag(false)}
            onDrop={(e) => { e.preventDefault(); setDrag(false); pick(e.dataTransfer.files[0]); }}
            className={`tw-flex tw-min-h-[220px] tw-cursor-pointer tw-items-center tw-justify-center tw-overflow-hidden tw-rounded-lg tw-border-2 tw-border-dashed tw-p-2 tw-text-center tw-transition ${
              drag ? "tw-border-qg-brand tw-bg-qg-panel2" : "tw-border-qg-line"}`}
          >
            {preview ? (
              <img src={preview} alt="Selected for inspection" className="tw-max-h-[300px] tw-max-w-full tw-rounded" />
            ) : (
              <span className="tw-text-qg-muted">
                Drag & drop an image here, or click to browse
                <br />
                <small>PNG · JPG · BMP · TIFF · WebP</small>
              </span>
            )}
          </div>
          <input ref={inputRef} type="file" accept={ACCEPT} hidden onChange={(e) => pick(e.target.files?.[0])} data-testid="inspect-file" />
          {file && <div className="tw-mt-2 tw-truncate tw-text-xs tw-text-qg-muted">{file.name} · {(file.size / 1024).toFixed(0)} kB</div>}

          <label className="tw-mt-4 tw-block tw-text-sm tw-font-medium tw-text-qg-ink" htmlFor="qg-category">2 · Component category</label>
          <select id="qg-category" value={profile} onChange={(e) => setProfile(e.target.value)} className="tw-mt-1 tw-w-full">
            <option value="auto">Auto-detect (recommended)</option>
            {profiles.map((p) => <option key={p.id} value={p.id}>{p.name}</option>)}
          </select>
          <p className="tw-mt-1 tw-text-xs tw-text-qg-muted">
            Auto-detect recognises the trained categories. Images of anything else are reported as unsupported and sent to manual inspection.
          </p>

          <button
            className="btn btn-primary btn-block tw-mt-4 tw-h-12 tw-text-base tw-font-bold tw-tracking-wider"
            disabled={!file || busy}
            onClick={run}
            data-testid="inspect-button"
          >
            {busy ? `ANALYSING… ${elapsed.toFixed(1)} s` : "INSPECT"}
          </button>
          {busy && (
            <div className="tw-mt-3 tw-h-1.5 tw-w-full tw-overflow-hidden tw-rounded tw-bg-qg-panel2" aria-label="Processing">
              <div className="tw-h-full tw-w-1/3 tw-animate-pulse tw-rounded tw-bg-qg-brand" />
            </div>
          )}
          {error && (
            <div role="alert" className="tw-mt-3 tw-rounded-lg tw-border tw-border-solid tw-border-verdict-defective tw-p-3 tw-text-sm tw-text-verdict-defective">
              {error}
            </div>
          )}
        </Panel>

        {/* ------------------------------------------------ result column */}
        <div className="tw-min-w-0">
          {result ? (
            <ResultView d={result} />
          ) : (
            <Panel>
              <div className="tw-py-16 tw-text-center tw-text-qg-muted">
                The inspection result, defect location, heat-maps and explanation will appear here.
                <div className="tw-mt-3 tw-text-xs">
                  Possible results: <b className="tw-text-verdict-normal">NORMAL</b> · <b className="tw-text-verdict-defective">DEFECTIVE</b> ·{" "}
                  <b className="tw-text-verdict-manual">MANUAL INSPECTION REQUIRED</b>
                </div>
              </div>
            </Panel>
          )}
        </div>
      </div>
    </div>
  );
}

function ResultView({ d }: { d: InspectionDetail }) {
  const r: InspectionResult = d.result;
  const v = VERDICT_STYLE[d.verdict];
  const clf = r.engines.classification;
  const an = r.engines.anomaly;
  const seg = r.engines.segmentation;
  const rec = r.recognition;
  const viz = d.visualizations || {};
  const [pdfBusy, setPdfBusy] = useState(false);

  const images = useMemo(() => {
    const out: { key: string; title: string; caption: string }[] = [];
    const add = (key: string, title: string, caption: string) => viz[key] && out.push({ key, title, caption });
    add("original", "Original image", "As uploaded (display-resized).");
    add("bounding_boxes", "Defect location", "Bounding boxes of the detected defect regions.");
    if (seg) add("segmentation_overlay", "Segmentation mask", "Supervised U-Net mask: the precise defect-boundary estimate.");
    else add("segmentation_overlay", "Estimated defect region", "Thresholded anomaly map (region estimate, not a measured boundary).");
    add("anomaly_patchcore_overlay", "Anomaly heat-map (PatchCore)", "Deviation from defect-free reference images.");
    add("anomaly_padim_overlay", "Anomaly heat-map (PaDiM)", "Second anomaly engine.");
    add("segmentation_unet_overlay", "Defect-probability map (U-Net)", "Per-pixel defect probability from the supervised model.");
    add("gradcam_overlay", "Grad-CAM explanation", "Where the classifier looked. An explanation, NOT a defect boundary.");
    add("gradcam_pp_overlay", "Grad-CAM++ explanation", "Alternative attribution. An explanation, NOT a defect boundary.");
    add("defect_zoom", "Suspicious region (crop)", "Magnified largest detected region.");
    return out;
  }, [viz, seg]);

  const anomalyRatio = an ? an.score_ratio[an.primary_engine] : null;

  return (
    <div className="tw-flex tw-flex-col tw-gap-4" data-testid="inspection-result">
      {/* verdict banner */}
      <section className={`tw-flex tw-flex-wrap tw-items-center tw-gap-4 tw-rounded-xl tw-border-2 tw-border-solid ${v.ring} tw-bg-qg-card tw-p-4`}>
        <div className={`tw-flex tw-h-14 tw-w-14 tw-shrink-0 tw-items-center tw-justify-center tw-rounded-full ${v.bg} tw-text-2xl tw-font-bold tw-text-white`} aria-hidden>
          {v.icon}
        </div>
        <div className="tw-min-w-0 tw-flex-1">
          <div className="tw-text-xs tw-uppercase tw-tracking-wider tw-text-qg-muted">Inspection result</div>
          <div className="tw-text-2xl tw-font-extrabold tw-tracking-wide tw-text-qg-ink" data-testid="verdict">{v.label}</div>
          <div className="tw-text-sm tw-text-qg-muted">{d.verdict_text}</div>
        </div>
        <div className="tw-flex tw-flex-wrap tw-gap-2">
          <button
            className="btn btn-primary btn-sm"
            disabled={pdfBusy}
            onClick={async () => {
              setPdfBusy(true);
              try { await download(`/api/inspections/${d.id}/report`, `QualityGuard_inspection_${d.id.slice(0, 8)}.pdf`); }
              finally { setPdfBusy(false); }
            }}
          >
            {pdfBusy ? "Generating PDF…" : "Download PDF report"}
          </button>
          <Link className="btn btn-ghost btn-sm" to={`/inspections/${d.id}`}>Full record</Link>
        </div>
      </section>

      {/* recognition / warnings */}
      {rec && (
        <div className={`tw-rounded-lg tw-border tw-border-solid tw-p-3 tw-text-sm ${
          rec.block ? "tw-border-verdict-manual tw-text-qg-ink" : "tw-border-qg-line tw-text-qg-muted"}`}>
          {rec.available === false ? (
            <span>Component recogniser unavailable — category was not checked. {rec.note}</span>
          ) : rec.block ? (
            <span><b className="tw-text-verdict-manual">⚠ {rec.block}</b></span>
          ) : rec.title ? (
            <span>
              Recognised category: <b className="tw-text-qg-ink">{rec.title}</b> ({pct(rec.probability)}; appearance distance{" "}
              {fix(rec.distance, 3)} ≤ limit {fix(rec.distance_threshold, 3)}) {rec.mode === "auto" ? "— selected automatically." : "— matches your selection."}
            </span>
          ) : (
            <span>{rec.note}</span>
          )}
        </div>
      )}

      {/* key numbers */}
      <Panel title="Assessment">
        <div className="tw-grid tw-grid-cols-2 tw-gap-3 md:tw-grid-cols-3 xl:tw-grid-cols-6">
          <Stat label="Defect category" value={d.defect_class ?? (r.defect_present ? "anomaly (unclassified)" : "—")}
            hint={clf ? undefined : "This category has no supervised defect-type model; anomalies are localised but not named."} />
          <Stat label="Classification confidence" value={clf ? pct(clf.confidence) : "n/a"}
            hint={clf ? `Temperature-calibrated; ${clf.members_agree ? "ensemble agrees" : "ensemble DISAGREES"}` : "No classifier for this category"} />
          <Stat label="Anomaly score" value={anomalyRatio != null ? `${fix(anomalyRatio)}× threshold` : "n/a"}
            hint={an ? `${an.primary_engine}; ≥ 1.00 means anomalous (threshold set on validation data)` : undefined} />
          <Stat label="Severity" value={<span className="tw-capitalize">{r.severity?.level ?? "—"}</span>}
            hint="Configurable rule-based estimate, not a physical measurement" />
          <Stat label="Image quality" value={r.input_valid ? "Suitable" : "Unsuitable"}
            hint={r.input_issues?.length ? r.input_issues.join(" ") : `sharpness ${fix(r.input?.sharpness_norm, 0)}, contrast ${fix(r.input?.contrast, 0)}`} />
          <Stat label="Inference time" value={`${(r.timings_ms?.total ?? d.latency_ms).toFixed(0)} ms`} hint="Measured on this server (CPU)" />
        </div>
        {seg && (
          <div className="tw-mt-3 tw-text-xs tw-text-qg-muted">
            Segmentation model: defect score {fix(seg.image_score, 3)} vs threshold {fix(seg.image_threshold, 3)} → {seg.is_defective ? "defect" : "no defect"}.
          </div>
        )}
        {r.fusion && (
          <div className="tw-mt-1 tw-text-xs tw-text-qg-muted">
            Engine votes: {Object.entries(r.fusion.votes).map(([k, val]) => `${k}: ${val ? "defect" : "no defect"}`).join(" · ")} — {r.fusion.rule}.
          </div>
        )}
      </Panel>

      {/* localisation + explanations */}
      {images.length > 0 && (
        <Panel title="Defect localisation & explanations">
          <div className="tw-grid tw-grid-cols-1 tw-gap-3 sm:tw-grid-cols-2 xl:tw-grid-cols-3">
            {images.map((im) => (
              <figure key={im.key} className="tw-m-0">
                <a href={authUrl(viz[im.key])} target="_blank" rel="noreferrer">
                  <img src={authUrl(viz[im.key])} alt={im.title} loading="lazy"
                    className="tw-w-full tw-rounded-lg tw-border tw-border-solid tw-border-qg-line tw-bg-black/5 tw-object-contain" style={{ maxHeight: 260 }} />
                </a>
                <figcaption className="tw-mt-1 tw-text-xs">
                  <b className="tw-text-qg-ink">{im.title}</b> <span className="tw-text-qg-muted">— {im.caption}</span>
                </figcaption>
              </figure>
            ))}
          </div>
          {r.measurements && r.measurements.num_regions > 0 && (
            <div className="tw-mt-3 tw-text-xs tw-text-qg-muted">
              {r.measurements.num_regions} region(s); defect area {pct(r.measurements.defect_area_ratio, 2)} of the image
              {r.measurements.calibrated ? "" : " (pixel units — no physical scale is calibrated for this category)"}.
            </div>
          )}
        </Panel>
      )}

      {/* why + recommendation */}
      <div className="tw-grid tw-gap-4 md:tw-grid-cols-2">
        <Panel title="Why this result">
          <ul className="tw-m-0 tw-list-disc tw-pl-5 tw-text-sm tw-text-qg-ink">
            {(r.decision.reasons || []).map((x, i) => <li key={i}>{x}</li>)}
            {(r.reliability?.notes || []).filter((x) => !r.decision.reasons?.includes(x)).map((x, i) => (
              <li key={`n${i}`} className="tw-text-qg-muted">{x}</li>
            ))}
            {(r.severity?.basis || []).map((x, i) => <li key={`s${i}`} className="tw-text-qg-muted">{x}</li>)}
          </ul>
        </Panel>
        <Panel title="Recommendation">
          <p className="tw-m-0 tw-text-sm tw-text-qg-ink">{r.decision.recommendation}</p>
          <p className="tw-mb-0 tw-mt-3 tw-text-xs tw-text-qg-muted">
            Line disposition by the "{d.profile_id}" profile rules: <b>{d.status}</b>. Models: {(r.models_used || []).join(", ") || "—"}.
          </p>
          <p className="tw-mb-0 tw-mt-2 tw-text-xs tw-text-qg-muted">
            Limitations: visual inspection only, within the categories and defect types these models were trained on. Internal
            faults (bearing wear, vibration, overheating, lubrication, electrical) cannot be diagnosed from a photograph.
          </p>
        </Panel>
      </div>
    </div>
  );
}
