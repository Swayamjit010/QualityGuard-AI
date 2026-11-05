// Typed facade over the JavaScript API helpers (src/api.js) plus the inspection result types.
import { api as apiJs, authUrl as authUrlJs, download as downloadJs } from "../api";

export interface ApiOptions {
  method?: string;
  body?: unknown;
  form?: FormData;
  query?: Record<string, string | number | boolean | undefined | null>;
  raw?: boolean;
}

export const api = apiJs as unknown as <T = unknown>(path: string, opts?: ApiOptions) => Promise<T>;
export const authUrl = authUrlJs as unknown as (path: string) => string;
export const download = downloadJs as unknown as (path: string, filename?: string) => Promise<void>;

export type Verdict = "NORMAL" | "DEFECTIVE" | "MANUAL INSPECTION REQUIRED";

export interface Recognition {
  mode: "auto" | "manual";
  available: boolean;
  domain?: string;
  title?: string;
  profile_id?: string;
  probability?: number;
  distance?: number;
  distance_threshold?: number;
  supported?: boolean;
  reasons?: string[];
  alternatives?: { domain: string; probability: number }[];
  block?: string;
  note?: string;
  latency_ms?: number;
}

export interface ClassificationEngine {
  predicted_class: string;
  confidence: number;
  margin: number;
  entropy: number;
  members_agree: boolean;
  unfamiliar: boolean;
  probabilities: Record<string, number>;
  ensemble: { run_id: string; arch: string; top_class: string; confidence: number; ood: boolean }[];
  explanations?: Record<string, { target_class: string; model: string }>;
}

export interface AnomalyEngine {
  model: string;
  scores: Record<string, number>;
  thresholds: Record<string, number>;
  score_ratio: Record<string, number>;
  primary_engine: string;
  is_anomalous: boolean;
  engines_disagree: boolean;
  defect_pixels_fraction: number;
}

export interface SegmentationEngine {
  model: string;
  image_score: number;
  image_threshold: number;
  score_ratio: number;
  is_defective: boolean;
  defect_pixels_fraction: number;
}

export interface Region {
  area_px: number;
  area_ratio: number;
  bbox_px: [number, number, number, number];
  area_mm2?: number;
}

export interface InspectionResult {
  profile_id: string;
  input: { width?: number; height?: number; brightness?: number; contrast?: number; sharpness?: number; sharpness_norm?: number };
  input_valid: boolean;
  input_issues: string[];
  engines: { classification?: ClassificationEngine; anomaly?: AnomalyEngine; segmentation?: SegmentationEngine;
    novelty?: { score: number; threshold: number; unfamiliar: boolean } };
  fusion?: { votes: Record<string, boolean>; rule: string; disagree: boolean };
  recognition?: Recognition;
  defect_present?: boolean;
  defect_class?: string | null;
  measurements?: { num_regions: number; defect_area_ratio: number; regions: Region[]; calibrated: boolean } | null;
  severity: { level: string; basis: string[] };
  decision: { status: string; verdict: Verdict; verdict_text: string; reasons: string[]; recommendation: string;
    manual_review_required: boolean; quality_index: number | null; message?: string };
  reliability?: { level: string; notes: string[] };
  unfamiliar?: boolean;
  models_used?: string[];
  timings_ms: Record<string, number>;
}

export interface InspectionDetail {
  id: string;
  created_at: string;
  profile_id: string;
  status: string;
  verdict: Verdict;
  verdict_text: string;
  defect_class: string | null;
  severity: string;
  confidence: number | null;
  anomaly_score_ratio: number | null;
  latency_ms: number;
  filename: string;
  result: InspectionResult;
  visualizations: Record<string, string>;
}

export interface Profile {
  id: string;
  name: string;
}

export const pct = (v?: number | null, d = 1) => (v == null ? "—" : `${(v * 100).toFixed(d)}%`);
export const fix = (v?: number | null, d = 2) => (v == null ? "—" : v.toFixed(d));
