import { useEffect, useMemo, useState } from "react";
import { convertFileSrc, invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { getCurrentWindow } from "@tauri-apps/api/window";
import { confirm as confirmDialog, open } from "@tauri-apps/plugin-dialog";
import {
  Activity,
  BarChart3,
  Check,
  CheckCircle2,
  ChevronDown,
  ChevronLeft,
	  ChevronRight,
  CircleDot,
  Cloud,
  FileText,
  FolderOpen,
  Gauge,
  History,
  Image as ImageIcon,
  LoaderCircle,
  Menu,
  MessageSquareText,
  Minus,
  Moon,
  PanelLeftClose,
  Play,
  Plus,
  RotateCcw,
  Settings,
  Sparkles,
  Square,
  Sun,
  Timer,
  Trash2,
  Video,
  WandSparkles,
  X,
  XCircle,
} from "lucide-react";
import type {
  AppSettings,
  PageRecord,
  StartTaskPayload,
  TaskRecord,
  WorkerEvent,
} from "./types";
import appLogo from "./assets/logo.png";

const DEFAULT_SETTINGS: AppSettings = {
  output_root: "",
  detector_preset: "precise",
  asr_engine: "mimo-cloud",
  asr_model: "small",
  asr_api_key: "",
  llm_api_key: "",
  mimo_base_url: "https://api.xiaomimimo.com/v1",
  mimo_model: "mimo-v2.5-asr",
  asr_concurrency: 3,
  llm_base_url: "https://api.xiaomimimo.com/v1/chat/completions",
  llm_model: "mimo-v2.5",
  llm_concurrency: 5,
  include_llm: true,
  include_evidence: false,
};

const MAX_SHARED_CLOUD_REQUESTS = 10;

function normalizeCloudConcurrency(value: AppSettings): AppSettings {
  let asrConcurrency = Math.min(
    MAX_SHARED_CLOUD_REQUESTS,
    Math.max(1, Number(value.asr_concurrency) || 1),
  );
  let llmConcurrency = Math.min(
    MAX_SHARED_CLOUD_REQUESTS,
    Math.max(1, Number(value.llm_concurrency) || 1),
  );
  if (value.asr_engine === "mimo-cloud" && value.include_llm) {
    asrConcurrency = Math.min(MAX_SHARED_CLOUD_REQUESTS - 1, asrConcurrency);
    llmConcurrency = Math.min(
      MAX_SHARED_CLOUD_REQUESTS - asrConcurrency,
      llmConcurrency,
    );
  }
  return {
    ...value,
    asr_concurrency: asrConcurrency,
    llm_concurrency: llmConcurrency,
  };
}


type NavKey = "tasks" | "reports" | "settings";

function formatTime(value?: number) {
  const seconds = Math.max(0, Number(value || 0));
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const rest = Math.floor(seconds % 60);
  return [hours, minutes, rest]
    .map((part) => String(part).padStart(2, "0"))
    .join(":");
}

function scoreLevel(score?: number) {
  if (score == null) return "等待评分";
  if (score >= 85) return "高度相关";
  if (score >= 70) return "明显相关";
  if (score >= 50) return "部分相关";
  return "关联较弱";
}

function statusLabel(status?: PageRecord["status"]) {
  return {
    waiting: "等待处理",
    detected: "页面已确认",
    transcribing: "语音识别中",
    scoring: "正在评分",
    completed: "已完成",
    failed: "处理失败",
  }[status || "waiting"];
}

function statusIcon(status?: PageRecord["status"]) {
  if (status === "completed") return <CheckCircle2 size={15} />;
  if (status === "failed") return <XCircle size={15} />;
  if (status === "transcribing") return <Activity size={15} />;
  if (status === "scoring") return <Sparkles size={15} />;
  if (status === "detected") return <Check size={15} />;
  return <Timer size={15} />;
}

function pageImage(page?: PageRecord) {
  return page?.screenshot_path ? convertFileSrc(page.screenshot_path) : "";
}

function overallScore(task?: TaskRecord) {
  const value = task?.summary?.strict_overall_score
    ?? task?.summary?.association_average_score;
  if (value == null) return undefined;
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : undefined;
}

function EmptySlide({ compact = false }: { compact?: boolean }) {
  return (
    <div className={`empty-slide ${compact ? "compact" : ""}`}>
      <div className="empty-slide-title">等待页面确认</div>
      <div className="empty-slide-line" />
      <div className="empty-slide-body">
        <span>页面确认后将自动加载截图</span>
      </div>
    </div>
  );
}

function SlideImage({
  page,
  compact = false,
}: {
  page?: PageRecord;
  compact?: boolean;
}) {
  const [failed, setFailed] = useState(false);
  const source = pageImage(page);
  useEffect(() => setFailed(false), [source]);
  if (!source || failed) return <EmptySlide compact={compact} />;
  return (
    <img
      className={compact ? "slide-image compact" : "slide-image"}
      src={source}
      alt={`第 ${page?.page_id || ""} 页 PPT`}
      onError={() => setFailed(true)}
    />
  );
}

function StagePipeline({ task }: { task?: TaskRecord }) {
  const stage = task?.stage || "";
  const pages = task?.pages || [];
  const isRunning = task?.status === "running";
  const isComplete = task?.status === "completed";
  const hasErrors = task?.status === "completed_with_errors";
  const isTerminal = isComplete || hasErrors;
  const voiceEnabled = task?.mode !== "detect";
  const llmEnabled = voiceEnabled && task?.include_llm !== false;

  const transcribed = pages.filter((p) => p.status === "scoring" || p.status === "completed" || p.status === "failed").length;
  const scored = pages.filter((p) => (p.status === "completed" || p.status === "failed") && p.score != null).length;
  const totalPages = pages.length;
  const completedStages = task?.completed_stages || [];

  const pptPct = isTerminal ? 100 : Math.min(100, Math.max(0, task?.stage_progresses?.ppt || 0));
  const voicePct = !voiceEnabled ? 0 : isTerminal ? 100 : totalPages ? Math.round((transcribed / totalPages) * 100) : 0;
  const llmPct = !llmEnabled ? 0 : isComplete ? 100 : totalPages ? Math.round((scored / totalPages) * 100) : 0;
  const stageRatio = isTerminal ? 100 : Math.round(pptPct * 0.45 + voicePct * 0.30 + llmPct * 0.25);

  // 并发流水线会在 PPT 检测尚未结束时提前完成个别页面。
  // 阶段只有在内核明确确认 PPT 完成，且对应页面进度达到 100% 后才允许打勾。
  const pptDone = isTerminal || (completedStages.includes("ppt") && pptPct === 100);
  const voiceDone = voiceEnabled && (isTerminal || (pptDone && totalPages > 0 && voicePct === 100));
  const llmDone = llmEnabled && !hasErrors && (
    isComplete || (pptDone && totalPages > 0 && llmPct === 100)
  );
  const reportDone = isTerminal;

  // Active states based on worker stage messages
  const pptActive = isRunning && !pptDone && (stage.includes("PPT") || stage.includes("页面") || totalPages === 0);
  const voiceActive = isRunning && !voiceDone && (stage.includes("语音") || stage.includes("转写") || stage.includes("云端"));
  const llmActive = isRunning && !llmDone && (stage.includes("LLM") || stage.includes("评分"));
  const reportActive = isRunning && !reportDone && (stage.includes("报告") || (!pptActive && !voiceActive && !llmActive && isRunning));

  const stages = [
    { label: "PPT识别", active: pptActive, done: pptDone, pct: pptPct, disabled: false, failed: false },
    { label: "语音识别", active: voiceActive, done: voiceDone, pct: voicePct, disabled: !voiceEnabled, failed: false },
    { label: "关联度评分", active: llmActive, done: llmDone, pct: llmPct, disabled: !llmEnabled, failed: hasErrors && llmEnabled },
    { label: "生成报告", active: reportActive, done: reportDone, pct: isTerminal ? 100 : stageRatio, disabled: false, failed: hasErrors },
  ];

  return (
    <div className="stage-pipeline">
      {stages.map(({ label, active, done, pct, disabled, failed }) => {
        const waiting = !disabled && !failed && !done && !active && isRunning;
        return (
          <div className="stage-wrap" key={label}>
            <div className={`stage-node ${done && !failed ? "done" : ""} ${active ? "active" : ""} ${waiting ? "waiting" : ""} ${failed ? "failed" : ""} ${disabled ? "disabled" : ""}`}>
              <span>{failed ? <X size={15} /> : done ? <Check size={15} /> : active ? <LoaderCircle size={15} /> : waiting ? <LoaderCircle size={15} /> : <span className="stage-dot" />}</span>
              <div>
                <strong>{label}</strong>
                <small>{disabled ? "未启用" : `${pct}%${failed ? " · 有错误" : done ? " · 已完成" : active ? " · 进行中" : waiting ? " · 等待中" : ""}`}</small>
              </div>
            </div>
            {label !== "生成报告" && <div className={`stage-link ${done && !failed ? "done" : ""}`} />}
          </div>
        );
      })}
    </div>
  );
}

function PageCard({
  page,
  selected,
  onClick,
}: {
  page: PageRecord;
  selected: boolean;
  onClick: () => void;
}) {
  return (
    <button className={`page-card ${selected ? "selected" : ""}`} onClick={onClick}>
      <div className="page-thumb">
        <SlideImage page={page} compact />
      </div>
      <div className="page-card-body">
        <div className="page-card-title">
          <strong>第 {page.page_id} 页</strong>
          <span>
            {formatTime(page.start_sec)} — {formatTime(page.end_sec)}
          </span>
        </div>
        <p>
          {page.speech_text?.trim()
            ? page.speech_text.trim().slice(0, 120) + (page.speech_text.trim().length > 120 ? "…" : "")
            : (page.status === "transcribing"
              ? "正在将本页时间范围内的课堂语音转换成文字…"
              : "本页暂时没有可显示的课堂转写内容。")}
        </p>
      </div>
      <div className="page-status-column">
        <span className={`status-chip ${page.status || "waiting"}`}>
          {statusIcon(page.status)}
          {statusLabel(page.status)}
        </span>
        {page.score != null ? (
          <div className="score-badge">
            <strong>{Math.round(page.score)}</strong>
          </div>
        ) : page.status === "scoring" ? (
          <div className="score-progress">
            <Sparkles size={19} />
            <span>分析中</span>
          </div>
        ) : page.status === "transcribing" ? (
          <div className="spinner-ring" />
        ) : null}
      </div>
    </button>
  );
}

function ReportViewer({
  task,
  onClose,
  onRetryPage,
  settings,
}: {
  task: TaskRecord;
  onClose: () => void;
  onRetryPage?: (page: PageRecord) => void;
  settings?: AppSettings;
}) {
  const score = overallScore(task);
  const coverage = Number(task.summary?.speech_page_coverage_percent ?? 0);
  const [pageIndex, setPageIndex] = useState(0);
  const evidenceEnabled = task.include_evidence ?? settings?.include_evidence ?? false;
  const pages = task.pages;
  const page = pages[pageIndex];
  const hasPrev = pageIndex > 0;
  const hasNext = pageIndex < pages.length - 1;
  const text = page?.speech_text?.trim() || "";
  const reason = page?.reason || "";
  return (
    <div className="modal-backdrop report-viewer-backdrop">
      <div className="modal report-viewer-modal">
        <div className="modal-header">
          <div>
            <h2>{task.video_id} — 分析报告</h2>
            <p>总关联度、逐页评分、PPT 截图、语音转录和评分理由都在这里汇总。</p>
          </div>
          <button className="icon-button" onClick={onClose}><X size={19} /></button>
        </div>
        <div className="report-viewer-summary">
          <article><Gauge /><span>总关联度</span><strong>{score != null ? Math.round(score) : "—"}</strong></article>
          <article><History /><span>讲话覆盖率</span><strong>{Math.round(coverage)}%</strong></article>
          <article><CheckCircle2 /><span>已完成页</span><strong>{pages.filter((p) => p.status === "completed").length}</strong></article>
          <article><FileText /><span>总页数</span><strong>{pages.length}</strong></article>
        </div>
        {page && (
          <div className="report-viewer-page-view">
            <div className="report-viewer-page-head">
              <strong>第 {page.page_id} 页</strong>
              <span>{formatTime(page.start_sec)} — {formatTime(page.end_sec)}</span>
              {page.score != null && <b>{Math.round(page.score)}分</b>}
              {page.status === "failed" && onRetryPage && (
                <button className="button retry-button" onClick={() => onRetryPage(page)}>
                  <RotateCcw size={15} />重试本页
                </button>
              )}
            </div>
            <div className="report-viewer-page-score">
              <i style={{ width: `${page.score || 0}%` }} />
            </div>
            <div className="report-viewer-page-body">
              <div className="report-viewer-shot">
                <SlideImage page={page} key={"r" + page.page_id + (page.screenshot_path || "")} />
              </div>
              <div className="report-viewer-text">
                <h4>原始转录</h4>
                <p>{text || "暂无语音识别内容。"}</p>
                <h4>评分理由</h4>
                <p>{reason || "暂无评分理由。"}</p>
                {page?.evidence != null && page.evidence.length > 0 && (page?.status === "completed" || page?.status === "failed") && (
                  <div className="evidence-content">
                    <h4>对应证据</h4>
                    {page.evidence.map((item, i) => (
                      <div key={i} className="evidence-item">
                        <div className="evidence-ppt"><b>PPT：</b>{item.ppt}</div>
                        <div className="evidence-speech"><b>讲话：</b>{item.speech}</div>
                      </div>
                    ))}
                  </div>
                )}
                {page?.evidence == null && (page?.status === "completed" || page?.status === "failed") && (
                  <div className="evidence-content">
                    <h4>对应证据</h4>
                    {evidenceEnabled ? (
                      <p>该页面没有匹配到对应证据。LLM 评估时可能未返回详细证据项。</p>
                    ) : (
                      <p>当前任务运行时的设置中「返回详细对应证据」为关闭状态，因此没有生成对应证据。请在设置中开启后，对<strong>新任务</strong>生效。</p>
                    )}
                  </div>
                )}
                {page?.evidence != null && page.evidence.length === 0 && (page?.status === "completed" || page?.status === "failed") && (
                  <div className="evidence-content">
                    <h4>对应证据</h4>
                    <p>该页面没有匹配到对应证据。</p>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}
        <div className="report-viewer-nav">
          <button className="button secondary" disabled={!hasPrev} onClick={() => setPageIndex((i) => i - 1)}>
            <ChevronLeft size={18} /> 上一页
          </button>
          <span className="report-viewer-page-counter">{pageIndex + 1} / {pages.length}</span>
          <button className="button secondary" disabled={!hasNext} onClick={() => setPageIndex((i) => i + 1)}>
            下一页 <ChevronRight size={18} />
          </button>
        </div>
      </div>
    </div>
  );
}
function Inspector({
  page,
  task,
  settings,
  onRetryPage,
}: {
  page?: PageRecord;
  task?: TaskRecord;
  settings?: AppSettings;
  onRetryPage?: (page: PageRecord) => void;
}) {
  const evidenceEnabled = task?.include_evidence ?? settings?.include_evidence ?? false;
  return (
    <aside className="inspector">
      <div className="inspector-header">
        <h2>{page ? `第 ${page.page_id} 页详情` : "页面详情"}</h2>
      </div>
      <div className="inspector-scroll">
        <div className="large-preview">
          <SlideImage page={page} />
        </div>
        <div className="time-range">
          <span>时间范围</span>
          <strong>
            {formatTime(page?.start_sec)} <i>—</i> {formatTime(page?.end_sec)}
          </strong>
        </div>
        <div className="divider" />
        <section className="score-section">
          <h3>关联度评分</h3>
          <div className="score-overview">
            <strong>{page?.score != null ? Math.round(page.score) : "—"}</strong>
            <div>
              <span className="score-level">
                <CircleDot size={13} />
                {scoreLevel(page?.score)}
              </span>
              <div className="score-line">
                <i style={{ width: `${page?.score || 0}%` }} />
              </div>
              <div className="score-scale">
                <span>0</span>
                <span>50</span>
                <span>100</span>
              </div>
            </div>
          </div>
        </section>
        <div className="divider" />
        <section className="reason-section">
          <h3>评分理由</h3>
          <p>
            {page?.reason ||
              (page?.status === "completed"
                ? "评分已完成，当前服务没有返回详细理由。"
                : "本页完成语音识别和关联度评分后，将在这里展示判断理由。")}
          </p>
          {page?.evidence != null && page.evidence.length > 0 && (page?.status === "completed" || page?.status === "failed") && (
            <div className="evidence-content" style={{ marginTop: 12 }}>
              <h4 style={{ margin: "0 0 8px", fontSize: 14, color: "var(--text)" }}>对应证据</h4>
              {page.evidence.map((item, i) => (
                <div key={i} className="evidence-item" style={{ marginBottom: 8 }}>
                  <div className="evidence-ppt"><b>PPT：</b>{item.ppt}</div>
                  <div className="evidence-speech"><b>讲话：</b>{item.speech}</div>
                </div>
              ))}
            </div>
          )}
          {page?.evidence == null && (page?.status === "completed" || page?.status === "failed") && (
            <div className="evidence-content" style={{ marginTop: 12 }}>
              <h4 style={{ margin: "0 0 8px", fontSize: 14, color: "var(--text)" }}>对应证据</h4>
              {evidenceEnabled ? (
                <p>该页面没有匹配到对应证据。LLM 评估时可能未返回详细证据项。</p>
              ) : (
                <p>当前任务运行时的设置中「返回详细对应证据」为关闭状态，因此没有生成对应证据。请在设置中开启后，对<strong>新任务</strong>生效。</p>
              )}
            </div>
          )}
          {page?.evidence != null && page.evidence.length === 0 && (page?.status === "completed" || page?.status === "failed") && (
            <div className="evidence-content" style={{ marginTop: 12 }}>
              <h4 style={{ margin: "0 0 8px", fontSize: 14, color: "var(--text)" }}>对应证据</h4>
              <p>该页面没有匹配到对应证据。</p>
            </div>
          )}
          {page?.status === "failed" && task?.status !== "running" && onRetryPage && (
            <button className="button retry-button inspector-retry" onClick={() => onRetryPage(page)}>
              <RotateCcw size={16} />
              {page.failure_stage === "asr" ? "重试本页语音识别" : "重试本页关联度评分"}
            </button>
          )}
        </section>
      </div>
    </aside>
  );
}

function SettingsModal({
  settings,
  onSave,
  onClose,
}: {
  settings: AppSettings;
  onSave: (value: AppSettings) => void;
  onClose: () => void;
}) {
  const [draft, setDraft] = useState(settings);
  const update = <K extends keyof AppSettings>(key: K, value: AppSettings[K]) =>
    setDraft((current) => normalizeCloudConcurrency({ ...current, [key]: value }));
  const asrConcurrencyMax = draft.include_llm
    ? MAX_SHARED_CLOUD_REQUESTS - 1
    : MAX_SHARED_CLOUD_REQUESTS;
  const llmConcurrencyMax = draft.asr_engine === "mimo-cloud" && draft.include_llm
    ? MAX_SHARED_CLOUD_REQUESTS - draft.asr_concurrency
    : MAX_SHARED_CLOUD_REQUESTS;
  return (
    <div className="modal-backdrop">
      <div className="modal settings-modal">
        <div className="modal-header">
          <div>
            <h2>模型与处理设置</h2>
            <p>API Key 仅保留在本次运行内存中，退出应用后不会保存。</p>
          </div>
          <button className="icon-button" onClick={onClose}><X size={19} /></button>
        </div>
        <div className="settings-grid">
          <section>
            <h3><Video size={17} /> PPT 页面识别</h3>
            <label>
              处理精度
              <select value={draft.detector_preset} onChange={(e) => update("detector_preset", e.target.value as "precise" | "fast")}>
                <option value="precise">智能精准（推荐）</option>
                <option value="fast">快速预览</option>
              </select>
            </label>
          </section>
          <section>
            <h3><MessageSquareText size={17} /> 语音识别</h3>
            <label>
              识别方式
              <select value={draft.asr_engine} onChange={(e) => update("asr_engine", e.target.value as AppSettings["asr_engine"])}>
                <option value="faster-whisper">本地 faster-whisper</option>
                <option value="mimo-cloud">ASR 模型服务（OpenAI 兼容）</option>
              </select>
            </label>
            {draft.asr_engine === "faster-whisper" ? (
              <label>本地模型<input value={draft.asr_model} onChange={(e) => update("asr_model", e.target.value)} /></label>
            ) : (
              <>
                <div className="form-two-columns aligned-fields">
                  <label className="wide-field">ASR 兼容地址<input value={draft.mimo_base_url} onChange={(e) => update("mimo_base_url", e.target.value)} /></label>
                  <label>模型名称<input value={draft.mimo_model} onChange={(e) => update("mimo_model", e.target.value)} /></label>
                  <label>
                    并发上限
                    <select value={draft.asr_concurrency} onChange={(e) => update("asr_concurrency", Number(e.target.value))}>
                    {Array.from({ length: asrConcurrencyMax }, (_, index) => index + 1).map((n) => <option key={n} value={n}>{n}</option>)}
                    </select>
                  </label>
                </div>
                <label>ASR API Key<input type="password" value={draft.asr_api_key} onChange={(e) => update("asr_api_key", e.target.value)} placeholder="sk-••••••••" /></label>
              </>
            )}
          </section>
          <section className="wide">
            <h3><WandSparkles size={17} /> LLM 关联度评分</h3>
            <div className="form-two-columns aligned-fields">
              <label>
                模型名称
                <input value={draft.llm_model} onChange={(e) => update("llm_model", e.target.value)} />
              </label>
              <label>
                并发上限
                <select value={draft.llm_concurrency} onChange={(e) => update("llm_concurrency", Number(e.target.value))}>
                  {Array.from({ length: llmConcurrencyMax }, (_, index) => index + 1).map((n) => <option key={n} value={n}>{n}</option>)}
                </select>
              </label>
              <label className="wide-field">OpenAI 兼容地址<input value={draft.llm_base_url} onChange={(e) => update("llm_base_url", e.target.value)} /></label>
            </div>
            <label>
              返回详细对应证据
              <select value={draft.include_evidence ? "true" : "false"} onChange={(e) => update("include_evidence", e.target.value === "true")}>
                <option value="false">关闭</option>
                <option value="true">开启</option>
              </select>
            </label>
            <label>LLM API Key<input type="password" value={draft.llm_api_key} onChange={(e) => update("llm_api_key", e.target.value)} placeholder="sk-••••••••" /></label>
          </section>
        </div>
        <div className="modal-actions">
          <button className="button secondary" onClick={onClose}>取消</button>
          <button className="button primary" onClick={() => onSave(draft)}>保存设置</button>
        </div>
      </div>
    </div>
  );
}

function NewTaskModal({
  settings,
  onClose,
  onStart,
}: {
  settings: AppSettings;
  onClose: () => void;
  onStart: (payload: StartTaskPayload) => Promise<void>;
}) {
  const [videoPath, setVideoPath] = useState("");
  const [outputRoot, setOutputRoot] = useState(settings.output_root);
  const [videoId, setVideoId] = useState("");
  const [mode, setMode] = useState<"full" | "detect">("full");
  const [error, setError] = useState("");

  async function chooseVideo() {
    const value = await open({
      multiple: false,
      filters: [{ name: "视频文件", extensions: ["mp4", "mkv", "mov", "avi", "wmv", "m4v", "webm"] }],
    });
    if (typeof value === "string") {
      setVideoPath(value);
      const file = value.split(/[\\/]/).pop() || "";
      setVideoId(file.replace(/\.[^.]+$/, ""));
    }
  }

  async function chooseOutput() {
    const value = await open({ directory: true, multiple: false });
    if (typeof value === "string") setOutputRoot(value);
  }

  async function submit() {
    if (!videoPath || !outputRoot || !videoId.trim()) {
      setError("请选择视频、结果目录并填写任务名称。");
      return;
    }
    await onStart({
      video_path: videoPath,
      output_root: outputRoot,
      video_id: videoId.trim(),
      mode,
      settings: {
        ...settings,
        asr_api_key: "",
        llm_api_key: "",
      },
      asr_api_key: settings.asr_api_key,
      llm_api_key: settings.llm_api_key,
      asr_upload_consent: true,
      llm_upload_consent: true,
    });
  }

  return (
    <div className="modal-backdrop">
      <div className="modal new-task-modal">
        <div className="modal-header">
          <div>
            <h2>新建课堂分析任务</h2>
            <p>选择一个课堂视频，课析会自动完成页面、语音和关联度分析。</p>
          </div>
          <button className="icon-button" onClick={onClose}><X size={19} /></button>
        </div>
        <div className="task-form">
          <label>
            课堂视频
            <button className="path-picker" onClick={chooseVideo}>
              <Video size={17} />
              <span>{videoPath || "选择 MP4 或其他课堂视频"}</span>
              <FolderOpen size={16} />
            </button>
          </label>
          <div className="form-two-columns">
            <label>任务名称<input value={videoId} onChange={(e) => setVideoId(e.target.value)} placeholder="例如：大学物理第3讲" /></label>
            <label>
              处理范围
              <select value={mode} onChange={(e) => setMode(e.target.value as "full" | "detect")}>
                <option value="full">完整流程：PPT + ASR + 评分</option>
                <option value="detect">仅识别 PPT 页面</option>
              </select>
            </label>
          </div>
          <label>
            结果目录
            <button className="path-picker" onClick={chooseOutput}>
              <FolderOpen size={17} />
              <span>{outputRoot || "选择结果保存目录"}</span>
              <ChevronDown size={16} />
            </button>
          </label>
          {error && <div className="form-error"><XCircle size={16} />{error}</div>}
        </div>
        <div className="modal-actions">
          <button className="button secondary" onClick={onClose}>取消</button>
          <button className="button primary" onClick={submit}><Play size={16} />开始处理</button>
        </div>
      </div>
    </div>
  );
}

function ReportsView({
  tasks,
  onOpenReport,
  onDelete,
  onRetryFailed,
}: {
  tasks: TaskRecord[];
  onOpenReport: (task: TaskRecord) => void;
  onDelete: (task: TaskRecord) => void;
  onRetryFailed: (task: TaskRecord) => void;
}) {
  const completed = tasks.filter((task) => task.status === "completed" || task.status === "completed_with_errors");
  const scores = completed
    .map((task) => overallScore(task))
    .filter((score): score is number => score != null);
  const average = scores.length ? scores.reduce((a, b) => a + b, 0) / scores.length : undefined;
  return (
    <main className="reports-view">
      <div className="reports-heading">
        <div><span>分析报告</span><h1>课堂质量概览</h1><p>汇总已经完成的课堂视频与 PPT 关联度分析结果。</p></div>
      </div>
      <div className="report-metrics">
        <article><History /><span>历史任务</span><strong>{tasks.length}</strong></article>
        <article><CheckCircle2 /><span>已完成任务</span><strong>{completed.length}</strong></article>
        <article><Gauge /><span>平均关联度</span><strong>{average != null ? Math.round(average) : "—"}</strong></article>
      </div>
      <section className="report-list">
        <h2>最近报告</h2>
        {completed.map((task) => {
          const score = overallScore(task);
          return (
            <div className="report-row" key={task.id}>
              <div className="report-icon"><FileText size={20} /></div>
              <div><strong>{task.video_id}</strong><span>{task.pages.length} 页 PPT · {task.model || "关联度模型"}</span></div>
              <div className="report-score">{score != null ? Math.round(score) : "—"}<small>分</small></div>
              <div className="report-actions">
                {task.status === "completed_with_errors" && (
                  <button className="button retry-button" onClick={() => onRetryFailed(task)}>
                    <RotateCcw size={15} />重试失败页
                  </button>
                )}
                <button className="button secondary" onClick={() => onOpenReport(task)}>打开报告</button>
                <button className="icon-button danger-button" title="删除报告及任务" onClick={() => onDelete(task)}>
                  <Trash2 size={16} />
                </button>
              </div>
            </div>
          );
        })}
        {!completed.length && <div className="empty-report">完成一次完整处理后，报告会显示在这里。</div>}
      </section>
    </main>
  );
}

export default function App() {
  const [tasks, setTasks] = useState<TaskRecord[]>([]);
  const [activeTaskId, setActiveTaskId] = useState("");
  const [selectedPageId, setSelectedPageId] = useState<number | null>(null);
  const [nav, setNav] = useState<NavKey>("tasks");
  const [settings, setSettings] = useState<AppSettings>(DEFAULT_SETTINGS);
  const [showSettings, setShowSettings] = useState(false);
  const [showNewTask, setShowNewTask] = useState(false);
  const [reportTask, setReportTask] = useState<TaskRecord | null>(null);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [cloudActive, setCloudActive] = useState(0);
  const [error, setError] = useState("");
  const [workerStatus, setWorkerStatus] = useState<"starting" | "ready" | "failed">("starting");
  const [algorithmVersion, setAlgorithmVersion] = useState("");
  const [theme, setTheme] = useState<"dark" | "light">(
    () => (localStorage.getItem("kexi.theme") as "dark" | "light") || "dark",
  );
  const activeTask = tasks.find((task) => task.id === activeTaskId) || tasks[0];
  const selectedPage =
    activeTask?.pages.find((page) => page.page_id === selectedPageId) ||
    activeTask?.pages[0];
  const completedPages = activeTask?.pages.filter((page) => page.status === "completed").length || 0;
  const failedPages = activeTask?.pages.filter((page) => page.status === "failed").length || 0;

  async function sendWorker(command: Record<string, unknown>) {
    await invoke("send_worker_command", { command });
  }

  useEffect(() => {
    if (!("__TAURI_INTERNALS__" in window)) {
      setWorkerStatus("failed");
      setError("请在课析桌面应用中运行，直接在浏览器中无法使用。");
      return;
    }
    const saved = localStorage.getItem("kexi.settings");
    let restored: Partial<AppSettings> = {};
    try {
      restored = saved ? JSON.parse(saved) : {};
    } catch {
      restored = {};
    }
    // 密钥只保留在本次应用内存中，不从旧版localStorage恢复。
    delete restored.asr_api_key;
    delete restored.llm_api_key;
    if (saved) {
      localStorage.setItem(
        "kexi.settings",
        JSON.stringify({
          ...restored,
          asr_api_key: "",
          llm_api_key: "",
        }),
      );
    }
    let unlisten: (() => void) | undefined;
    let disposed = false;
    listen<WorkerEvent & { tasks?: TaskRecord[] }>("worker-event", (event) => {
      const data = event.payload;
      if (data.type === "worker.ready") {
        setWorkerStatus("ready");
        setAlgorithmVersion(data.algorithm_version || "");
        return;
      }
      if (data.type === "worker.error" || data.type === "worker.exited") {
        const message = data.error || "后台处理引擎已经停止。";
        setWorkerStatus("failed");
        setError(message);
        setCloudActive(0);
        setTasks((current) => current.map((task) =>
          task.status === "running"
            ? { ...task, status: "failed", stage: "处理引擎异常", message }
            : task,
        ));
        return;
      }
      if (data.type === "tasks.list" && data.tasks) {
        const filtered = data.tasks.filter((t) => t.pages.length > 0 || t.status === "running");
        setTasks(filtered);
        setActiveTaskId((current) => current || filtered[0]?.id || "");
        return;
      }
      if (data.type === "task.started") {
        setCloudActive(0);
        const task: TaskRecord = {
          id: data.task_id || `task-${Date.now()}`,
          video_id: (data as WorkerEvent & { video_id?: string }).video_id || "新任务",
          video_path: (data as WorkerEvent & { video_path?: string }).video_path,
          run_dir: (data as WorkerEvent & { run_dir?: string }).run_dir || "",
          status: "running",
          progress: 0,
          stage: "准备处理",
          mode: data.mode,
          include_llm: data.include_llm,
          include_evidence: data.include_evidence ?? false,
          pages: [],
        };
        setTasks((current) => [task, ...current.filter((item) => item.id !== task.id)]);
        setActiveTaskId(task.id);
        setSelectedPageId(null);
        setError("");
      } else if (data.type === "cloud.activity") {
        setCloudActive(Math.max(0, Number(data.active_cloud_requests) || 0));
      } else if (data.type === "task.progress") {
        setTasks((current) =>
          current.map((task) =>
            task.id === (data.task_id || activeTaskId) ? (() => {
              const stageProgresses = { ...task.stage_progresses };
              if (data.stage_progress != null) {
                if (data.stage?.includes("PPT") || data.stage?.includes("页面识别")) stageProgresses.ppt = data.stage_progress;
                else if (data.stage?.includes("语音") || data.stage?.includes("转写")) stageProgresses.voice = data.stage_progress;
                else if (data.stage?.includes("LLM") || data.stage?.includes("评分")) stageProgresses.llm = data.stage_progress;
              }
              const completedStages = [...(task.completed_stages || [])];
              if (data.completed_stage && !completedStages.includes(data.completed_stage)) completedStages.push(data.completed_stage);
              return {
                ...task,
                status: "running" as const,
                progress: data.progress ?? task.progress,
                stage: data.stage,
                stage_progress: data.stage_progress ?? task.stage_progress,
                stage_progresses: stageProgresses,
                completed_stages: completedStages,
                message: data.message,
              };
            })() : task,
          ),
        );
      } else if (data.type === "page.updated" && data.page) {
        setTasks((current) =>
          current.map((task) => {
            if (data.task_id && task.id !== data.task_id) return task;
            if (task.status !== "running") return task;
            const exists = task.pages.some((page) => page.page_id === data.page!.page_id);
            const pages = exists
              ? task.pages.map((page) => {
                if (page.page_id !== data.page!.page_id) return page;
                const merged = { ...page, ...data.page };
                if (data.page!.status === "scoring" || data.page!.status === "completed") {
                  delete merged.failure_stage;
                }
                return merged;
              })
              : [...task.pages, data.page!].sort((a, b) => a.page_id - b.page_id);
            return { ...task, pages };
          }),
        );
        setSelectedPageId((current) => current ?? data.page!.page_id);
      } else if (data.type === "task.completed" && data.result) {
        const result = { ...data.result, elapsed_sec: data.elapsed_sec ?? data.result.elapsed_sec };
        setTasks((current) => [result, ...current.filter((task) => task.id !== result.id)]);
        setActiveTaskId(result.id);
        setCloudActive(0);
      } else if (data.type === "task.failed") {
        setError(data.error || "任务处理失败。");
        setCloudActive(0);
        setTasks((current) => current.map((task) => (task.id === (data.task_id || task.id) && task.status === "running") ? { ...task, status: "failed", stage: "处理失败" } : task));
      } else if (data.type === "task.retry_started") {
        const retryIds = new Set(data.page_ids || []);
        setError("");
        setTasks((current) => current.map((task) => task.id === data.task_id ? {
          ...task,
          status: "running",
          stage: "正在重试失败页",
          message: `正在重试 ${retryIds.size} 个失败页`,
          pages: task.pages.map((page) => retryIds.has(page.page_id) ? {
            ...page,
            status: page.failure_stage === "asr" ? "transcribing" : "scoring",
          } : page),
        } : task));
      } else if (data.type === "task.retry_completed" && data.result) {
        const result = data.result;
        setTasks((current) => [result, ...current.filter((task) => task.id !== result.id)]);
        setActiveTaskId(result.id);
        setReportTask((current) => current?.id === result.id ? result : current);
        setCloudActive(0);
      } else if (data.type === "task.retry_failed") {
        setError(data.error || "失败页重试未能完成。");
        setCloudActive(0);
        setTasks((current) => current.map((task) => task.id === data.task_id ? {
          ...task,
          status: "completed_with_errors",
          stage: "重试失败",
          message: data.error || "失败页重试未能完成。",
        } : task));
      } else if (data.type === "task.deleted") {
        setTasks((current) => current.filter((task) => task.id !== data.task_id));
        setActiveTaskId((active) => active === data.task_id ? "" : active);
        setSelectedPageId(null);
        setReportTask((current) => current?.id === data.task_id ? null : current);
      } else if (data.type === "task.delete_failed") {
        setError(data.error || "删除任务失败。");
      }
    }).then(async (dispose) => {
      if (disposed) {
        dispose();
        return;
      }
      unlisten = dispose;
      const output = await invoke<string>("project_output_dir");
      const merged = { ...DEFAULT_SETTINGS, ...restored };
      if (!merged.output_root) merged.output_root = output;
      setSettings(normalizeCloudConcurrency(merged));
      await sendWorker({ action: "ping" });
      await sendWorker({ action: "list_tasks", output_root: merged.output_root });
    }).catch((reason) => {
      setWorkerStatus("failed");
      setError(`无法初始化处理引擎：${String(reason)}`);
    });
    return () => {
      disposed = true;
      unlisten?.();
    };
  }, []);

  useEffect(() => {
    if (!activeTask || activeTask.status !== "running") return;
    const id = activeTaskId;
    const timer = window.setInterval(() => {
      setTasks((current) => current.map((t) =>
        t.id === id ? { ...t, elapsed_sec: (t.elapsed_sec || 0) + 1 } : t,
      ));
    }, 1000);
    return () => window.clearInterval(timer);
  }, [activeTaskId, activeTask?.status]);

  useEffect(() => {
    if (!activeTask) return;
    const preferred = [...activeTask.pages].reverse().find((page) => page.status === "completed") || activeTask.pages[0];
    if (preferred && !activeTask.pages.some((page) => page.page_id === selectedPageId)) {
      setSelectedPageId(preferred.page_id);
    }
  }, [activeTaskId, activeTask?.pages.length]);

  async function startTask(payload: StartTaskPayload) {
    setShowNewTask(false);
    setNav("tasks");
    setError("");
    try {
      await sendWorker({ action: "start", payload });
    } catch (reason) {
      setWorkerStatus("failed");
      setError(`无法启动任务：${String(reason)}`);
    }
  }

  async function saveSettings(value: AppSettings) {
    const normalized = normalizeCloudConcurrency(value);
    setSettings(normalized);
    localStorage.setItem(
      "kexi.settings",
      JSON.stringify({
        ...normalized,
        asr_api_key: "",
        llm_api_key: "",
      }),
    );
    setShowSettings(false);
    try {
      await sendWorker({ action: "list_tasks", output_root: normalized.output_root });
    } catch (reason) {
      setWorkerStatus("failed");
      setError(`设置已保存，但无法刷新任务列表：${String(reason)}`);
    }
  }

  async function deleteTask(task: TaskRecord) {
    if (task.status === "running") {
      setError("正在处理或重试的任务不能删除。");
      return;
    }
    const confirmed = await confirmDialog(
      `确定删除“${task.video_id}”吗？任务结果、转写和分析报告都会一并删除，且无法恢复。`,
      { title: "删除任务", kind: "warning", okLabel: "删除", cancelLabel: "取消" },
    );
    if (!confirmed) return;
    setError("");
    try {
      await sendWorker({
        action: "delete_task",
        task_id: task.id,
        output_root: settings.output_root,
      });
    } catch (reason) {
      setError(`无法删除任务：${String(reason)}`);
    }
  }

  async function retryFailedPages(task: TaskRecord, pageIds?: number[]) {
    if (tasks.some((item) => item.status === "running")) {
      setError("已有任务正在处理，请完成后再重试失败页。");
      return;
    }
    const failed = task.pages.filter((page) => page.status === "failed");
    const selected = pageIds?.length
      ? failed.filter((page) => pageIds.includes(page.page_id))
      : failed;
    if (!selected.length) {
      setError("没有找到可重试的失败页面。");
      return;
    }
    const asrCount = selected.filter((page) => page.failure_stage === "asr").length;
    const llmCount = selected.length - asrCount;
    const description = [
      asrCount ? `${asrCount} 页语音识别` : "",
      llmCount ? `${llmCount} 页关联度评分` : "",
    ].filter(Boolean).join("、");
    const confirmed = await confirmDialog(
      `将重试${description}。重试时只会重新发送这些失败页所需的音频、PPT 截图和转写内容，是否继续？`,
      { title: "重试失败页", kind: "warning", okLabel: "继续重试", cancelLabel: "取消" },
    );
    if (!confirmed) return;
    setError("");
    setNav("tasks");
    setActiveTaskId(task.id);
    try {
      await sendWorker({
        action: "retry_failed_pages",
        payload: {
          task_id: task.id,
          output_root: settings.output_root,
          page_ids: pageIds,
          settings,
          asr_api_key: settings.asr_api_key,
          llm_api_key: settings.llm_api_key,
          asr_upload_consent: true,
          llm_upload_consent: true,
        },
      });
    } catch (reason) {
      setError(`无法启动失败页重试：${String(reason)}`);
    }
  }

  const recentTasks = useMemo(() => tasks.filter((t) => t.pages.length > 0 || t.status === "running").slice(0, 8), [tasks]);

  return (
    <div className={`app-shell ${sidebarCollapsed ? "sidebar-collapsed" : ""} ${theme}`}>
      <header
        className="titlebar"
        data-tauri-drag-region
        onMouseDown={(event) => {
          if (event.button !== 0) return;
          if ((event.target as HTMLElement).closest("button")) return;
          void getCurrentWindow().startDragging();
        }}
      >
        <div className="brand" data-tauri-drag-region>
          <img className="brand-logo" src={appLogo} alt="课析" />
          <strong>课析</strong>
        </div>
        <div className="project-switcher" data-tauri-drag-region>
          <span>{activeTask?.video_id || "尚未选择任务"}</span>
          <ChevronDown size={15} />
        </div>
        <div className="titlebar-spacer" data-tauri-drag-region />
        <div className={`cloud-health ${workerStatus === "failed" ? "has-error" : ""}`}>
          <Cloud size={16} />
          {workerStatus === "ready"
            ? `处理引擎正常${algorithmVersion ? ` · 内核 ${algorithmVersion}` : ""}`
            : workerStatus === "failed"
              ? "处理引擎异常"
              : "正在连接处理引擎"}
        </div>
        <div className="window-actions">
          <button onClick={() => { const t = theme === "dark" ? "light" : "dark"; setTheme(t); localStorage.setItem("kexi.theme", t); }} title="切换主题">
            {theme === "dark" ? <Sun size={17} /> : <Moon size={17} />}
          </button>
          <button onClick={() => getCurrentWindow().minimize()}><Minus size={17} /></button>
          <button onClick={() => getCurrentWindow().toggleMaximize()}><Square size={13} /></button>
          <button className="close" onClick={() => getCurrentWindow().close()}><X size={18} /></button>
        </div>
      </header>

      <aside className="sidebar">
        <button className="new-task-button" onClick={() => setShowNewTask(true)}>
          <Plus size={19} /><span>新建任务</span>
        </button>
        <nav>
          <button className={nav === "tasks" ? "active" : ""} onClick={() => setNav("tasks")}><History size={19} /><span>任务中心</span></button>
          <button className={nav === "reports" ? "active" : ""} onClick={() => setNav("reports")}><BarChart3 size={19} /><span>分析报告</span></button>
          <button onClick={() => setShowSettings(true)}><Settings size={19} /><span>设置</span></button>
        </nav>
        <div className="recent-header"><span>最近任务</span><ChevronDown size={14} /></div>
        <div className="recent-list">
          {recentTasks.map((task) => (
            <div className="recent-task-row" key={task.id}>
              <button
                className={`recent-task-main ${task.id === activeTask?.id ? "active" : ""}`}
                onClick={() => {
                  setActiveTaskId(task.id);
                  setNav("tasks");
                }}
              >
                <div className="recent-thumb">
                  {task.pages[0] ? <SlideImage page={task.pages[0]} compact /> : <ImageIcon size={18} />}
                </div>
                <div>
                  <strong>{task.video_id}</strong>
                  <span>
                    {task.status === "running"
                      ? "正在处理"
                      : task.status === "completed_with_errors"
                        ? `${task.pages.length} 页 · 存在错误`
                        : `${task.pages.length} 页`}
                    {task.elapsed_sec != null && task.elapsed_sec > 0
                      ? ` · ${formatTime(task.elapsed_sec)}`
                      : ""}
                  </span>
                </div>
                {task.status === "running" && <i className="task-dot" />}
              </button>
              {task.status !== "running" && (
                <button className="recent-delete" title="删除任务" onClick={() => void deleteTask(task)}>
                  <Trash2 size={14} />
                </button>
              )}
            </div>
          ))}
          {!recentTasks.length && <div className="no-recent">还没有处理记录</div>}
        </div>
        <button className="collapse-button" onClick={() => setSidebarCollapsed((value) => !value)}>
          {sidebarCollapsed ? <Menu size={17} /> : <><PanelLeftClose size={17} /><span>收起侧栏</span></>}
        </button>
      </aside>

      {nav === "reports" ? (
        <ReportsView
          tasks={tasks}
          onOpenReport={(t) => setReportTask(t)}
          onDelete={(t) => void deleteTask(t)}
          onRetryFailed={(t) => void retryFailedPages(t)}
        />
      ) : (
        <>
          <main className="workspace">
            <section className="workspace-header">
              <div className="heading">
                <span>实时任务工作台</span>
                <h1>课堂视频分析</h1>
              </div>
              {activeTask && (
                <div className="progress-summary">
                  <span>处理中</span>
                  <strong>{completedPages}</strong>
                  <span>/ {activeTask.pages.length || "—"} 页</span>
                  {activeTask.summary?.strict_overall_score != null && (
                    <><i /><Gauge size={16} /><strong>{Math.round(activeTask.summary.strict_overall_score)}</strong><span>总关联度</span></>
                  )}
                  {activeTask.summary?.association_average_score != null && (
                    <><i /><strong>{Math.round(activeTask.summary.association_average_score)}</strong><span>平均分</span></>
                  )}
                  {activeTask.summary?.speech_page_coverage_percent != null && (
                    <><i /><strong>{Math.round(activeTask.summary.speech_page_coverage_percent)}%</strong><span>覆盖率</span></>
                  )}
                  <i />
                  <Timer size={17} />
                  <span>{formatTime(activeTask?.elapsed_sec || 0)}</span>
                </div>
              )}
              <StagePipeline task={activeTask} />
            </section>

            {!activeTask ? (
              <div className="welcome-empty">
                <div className="welcome-icon"><Video size={34} /></div>
                <span>从一个课堂视频开始</span>
                <h2>识别 PPT、转换讲话并完成关联度评分</h2>
                <p>课析会以实时流水线显示每一页的处理状态和最终结果。</p>
                <button className="button primary" onClick={() => setShowNewTask(true)}><Plus size={17} />新建分析任务</button>
                {error && <div className="task-error"><XCircle size={17} /><span>{error}</span><button onClick={() => setError("")}>关闭</button></div>}
              </div>
            ) : (
              <section className="page-stream">
                <div className="stream-toolbar">
                  <div>
                    <Activity size={16} />
                    <span>{activeTask.message || activeTask.stage || "已加载历史处理结果"}</span>
                  </div>
                  {failedPages > 0 && activeTask.status !== "running" && (
                    <button className="retry-button" onClick={() => void retryFailedPages(activeTask)}>
                      <RotateCcw size={15} />一键重试 {failedPages} 个失败页
                    </button>
                  )}
                </div>
                {error && <div className="task-error"><XCircle size={17} /><span>{error}</span><button onClick={() => setError("")}>关闭</button></div>}
                <div className="page-list">
                  {activeTask.pages.map((page) => (
                    <PageCard
                      key={page.page_id}
                      page={page}
                      selected={page.page_id === selectedPage?.page_id}
                      onClick={() => setSelectedPageId(page.page_id)}
                    />
                  ))}
                  {activeTask.status === "running" && !activeTask.pages.length && (
                    <div className="processing-placeholder">
                      <LoaderCircle size={26} />
                      <strong>正在分析视频时间线</strong>
                      <span>确认第一页后会立即显示在这里</span>
                    </div>
                  )}
                </div>
              </section>
            )}
          </main>
          <Inspector
            page={selectedPage}
            task={activeTask}
            settings={settings}
            onRetryPage={(page) => activeTask && void retryFailedPages(activeTask, [page.page_id])}
          />
        </>
      )}

      <footer className="statusbar">
        <div><Cloud size={15} /><span>当前云端并发</span><strong>{cloudActive}</strong></div>
        <div><CheckCircle2 size={16} /><span>已完成</span><strong>{completedPages} 页</strong></div>
        <div className={failedPages ? "has-error" : ""}><XCircle size={16} /><strong>{failedPages}</strong><span>个错误</span></div>
        <div className="status-spacer" />
        <div><span>模型</span><strong className="model-name">{settings.llm_model || "MiMo"}</strong></div>
        <div><Activity size={16} className="pulse" /><span>{activeTask?.status === "running" ? "服务活动中" : "服务待命"}</span></div>
      </footer>

      {reportTask && (
        <ReportViewer
          task={reportTask}
          settings={settings}
          onClose={() => setReportTask(null)}
          onRetryPage={(page) => void retryFailedPages(reportTask, [page.page_id])}
        />
      )}
      {showSettings && <SettingsModal settings={settings} onSave={saveSettings} onClose={() => setShowSettings(false)} />}
      {showNewTask && <NewTaskModal settings={settings} onStart={startTask} onClose={() => setShowNewTask(false)} />}
    </div>
  );
}
