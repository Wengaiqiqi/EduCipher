export type PageStatus =
  | "waiting"
  | "detected"
  | "transcribing"
  | "scoring"
  | "completed"
  | "failed";

export interface PageRecord {
  page_id: number;
  start_sec: number;
  end_sec: number;
  screenshot_path?: string;
  speech_text?: string;
  score?: number;
  level?: string;
  reason?: string;
  evidence?: Array<{ ppt: string; speech: string }>;
  status?: PageStatus;
  confidence?: string;
}

export interface TaskRecord {
  id: string;
  video_id: string;
  video_path?: string;
  run_dir: string;
  updated_at?: number;
  status: "idle" | "running" | "completed" | "failed";
  progress?: number;
  stage?: string;
  stage_progress?: number;
  message?: string;
  elapsed_sec?: number;
  model?: string;
  include_evidence?: boolean;
  summary?: {
    strict_overall_score?: number;
    association_average_score?: number;
    speech_page_coverage_percent?: number;
    total_pages?: number;
    scored_pages?: number;
  };
  pages: PageRecord[];
}

export interface AppSettings {
  output_root: string;
  detector_preset: "precise" | "fast";
  asr_engine: "mimo-cloud" | "faster-whisper";
  asr_model: string;
  asr_api_key: string;
  llm_api_key: string;
  mimo_base_url: string;
  mimo_model: string;
  asr_concurrency: number;
  llm_base_url: string;
  llm_model: string;
  llm_concurrency: number;
  include_llm: boolean;
  include_evidence: boolean;
}

export interface StartTaskPayload {
  video_path: string;
  output_root: string;
  video_id: string;
  mode: "full" | "detect";
  settings: AppSettings;
  asr_api_key?: string;
  llm_api_key?: string;
  asr_upload_consent: boolean;
  llm_upload_consent: boolean;
}

export interface WorkerEvent {
  type: string;
  task_id?: string;
  stage?: string;
  message?: string;
  progress?: number;
  stage_progress?: number;
  page?: PageRecord;
  pages?: PageRecord[];
  result?: TaskRecord;
  error?: string;
  traceback?: string;
  elapsed_sec?: number;
  active_cloud_requests?: number;
  cloud_limit?: number;
  algorithm_version?: string;
  include_evidence?: boolean;
}
