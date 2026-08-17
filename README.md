# 课析

课堂视频 PPT 识别、语音转写与讲授内容关联度分析桌面应用。

![课析 Logo](desktop_v2/src/assets/logo.png)

课析从课堂录像中识别 PPT 页面及其出现时间，按页整理讲话文字，再通过 OpenAI 兼容的多模态模型评估讲话与 PPT 的关联程度。当前桌面端版本为 **2.0.26**，分析内核版本为 **1.4.17**。

## 宣传片

<video src="https://github.com/user-attachments/assets/48317655-d435-483a-84b5-9debfe7256be" controls width="720"></video>

## 主要功能

- 自动识别 PPT 页面、起止时间和原视频分辨率截图。
- 提供“智能时序算法”和“FFmpeg 场景阈值算法”两种页面识别方式。
- 支持本地 faster-whisper 或 OpenAI 兼容的云端 ASR 服务。
- 将讲话按 PPT 页面时间段归类，不保留中间音频文件。
- 使用 OpenAI 兼容的多模态 LLM 逐页评分，可选返回详细对应证据。
- 支持一次添加多个视频；任务按阶段流水处理，空闲的 PPT、ASR 或 LLM 阶段会继续处理后续任务。
- 支持单页重试、一键重试失败页、删除误识别页面并自动重排页码。
- 持久保存最近任务、耗时、处理进度和分析报告。
- 提供亮色/暗色主题及 Windows 安装包。

## 处理流程

```text
课堂视频
   ↓
PPT 页面识别与时间轴
   ↓
按页面时间范围进行语音识别
   ↓
PPT 截图 + 逐页讲话文字发送给多模态 LLM
   ↓
逐页评分、总分与分析报告
```

多任务并非等待前一个任务全部结束后才开始。当前任务完成 PPT 识别后，PPT 识别器会立即处理队列中的下一个视频；前一个任务的 ASR、LLM 评分和报告生成继续运行。云端 ASR 与 LLM 使用全局并发限制，避免多个任务叠加后超过服务限流。

## 使用桌面应用

1. 安装并打开“课析”。首次启动默认使用亮色主题。
2. 在“设置”中选择结果目录、PPT 算法、语音识别方式和 LLM 服务。
3. 点击“新建任务”，可一次选择一个或多个课堂视频。
4. 选择“完整流程”或“仅识别 PPT”，然后开始处理。
5. 在“任务中心”查看逐页进度，在“分析报告”中查看最终结果。

任务失败时可以重试单页或一键重试全部失败页。误识别的 PPT 页面可通过页面右键菜单直接删除；删除后会重新编号，并从后续转写和评分流程中排除。

### PPT 页面识别算法

| 算法 | 适用场景 |
|---|---|
| 智能时序算法（推荐） | 适合大多数课堂录屏，可合并同页动画并精修换页时间 |
| 场景阈值算法（FFmpeg） | 适合画面切换明确、需要快速粗筛的录像 |

时序分析使用低分辨率帧提高速度，最终页面截图会从原视频重新提取。默认不裁切最终截图，避免投影边缘内容丢失。

### 语音识别

支持两种方式：

- **本地 faster-whisper**：选择包含 `model.bin` 的完整模型权重文件夹，不需要上传音频。
- **云端 ASR**：填写 OpenAI 兼容地址、模型名称、API Key 和并发上限。默认配置适配小米 MiMo `mimo-v2.5-asr`。

程序仅临时提取当前页面所需的音频片段，请求结束后立即清理，不保留 WAV 或 MP3 文件。第一版保留视频中识别到的全部讲话，不区分教师和学生。

### LLM 关联度评分

填写支持图片输入的 OpenAI 兼容接口、模型名称和 API Key。每页请求包含：

- 当前 PPT 截图；
- 该页面时间范围内的讲话文字；
- 统一的评分规则。

讲话文字不携带逐句时间戳，以减少输入 Token。开启“返回详细对应证据”后，报告会额外显示 PPT 内容与讲话内容的对应项。

汇总字段以以下三个结果为准：

- `strict_overall_score`：所有 PPT 页的严格总分；
- `association_average_score`：有讲话页面的关联度平均分；
- `speech_page_coverage_percent`：有讲话页面覆盖率。

API Key 只保留在当前应用内存中，不写入任务结果、设置文件或日志。修改设置不会中断正在运行的任务，新设置从后续任务开始生效。

## 输出目录

每个任务写入 `<结果目录>/<任务名称>/`：

```text
<任务名称>/
├─ page_001.jpg
├─ page_002.jpg
├─ result.json
├─ run_metadata.json
├─ transcript.json
├─ 逐页语音文字.md
└─ llm_evaluation/
   ├─ pages/
   │  ├─ page_001.json
   │  └─ page_NNN.json
   ├─ llm_evaluation.json
   └─ PPT讲话关联度报告.md
```

核心文件：

- `result.json`：PPT 页码、起止时间、代表帧和截图信息；
- `transcript.json`：逐页讲话文字；
- `llm_evaluation.json`：逐页评分和汇总结果；
- `PPT讲话关联度报告.md`：可直接阅读的最终报告。

## 从源码运行

### 环境要求

- Windows 10/11；
- Python 3.10 或更高版本；
- Node.js；
- Rust stable、Microsoft C++ Build Tools 和 WebView2；
- `ffmpeg` 与 `ffprobe` 已加入 `PATH`。

安装 Python 和前端依赖：

```powershell
python -m pip install -e .
python -m pip install pyinstaller
cd desktop_v2
npm install
```

启动 Tauri 桌面开发版：

```powershell
cd desktop_v2
npm run dev
```

也可以启动 Python 统一桌面入口：

```powershell
python -m video_page_detector desktop-gui
```

## 构建 Windows 安装包

```powershell
cd desktop_v2
npm run dist:win
```

该命令先使用 PyInstaller 打包 Python 处理内核，再生成 Tauri NSIS 安装包。构建结果位于：

```text
desktop_v2/src-tauri/target/release/bundle/nsis/
```

## 命令行

查看所有命令：

```powershell
python -m video_page_detector --help
```

只识别 PPT 页面：

```powershell
python -m video_page_detector detect "E:\课堂视频\lesson.mp4" `
  --config "config\default.json" `
  --output "E:\课析结果"
```

其他入口包括 `transcribe`、`evaluate-llm`、`gui`、`transcribe-gui` 和 `llm-evaluation-gui`。日常使用推荐直接使用 Tauri 桌面端。

## 测试

主程序：

```powershell
python -m unittest discover -s tests -v
```

归档场景阈值算法：

```powershell
python -m unittest discover -s "场景阈值方法/legacy_ffmpeg_scene_detector/tests" -v
```

前端类型检查与构建：

```powershell
cd desktop_v2
npm run build
```

## 项目结构

```text
desktop_v2/                         Tauri + React + TypeScript 桌面端
video_page_detector/                PPT、ASR、LLM 与任务流水线内核
config/                             默认检测、转写和评分配置
tests/                              Python 主流程测试
场景阈值方法/legacy_ffmpeg_scene_detector/  场景阈值算法实现
```

主要入口：

- `desktop_v2/src/App.tsx`：任务中心、设置和分析报告；
- `desktop_v2/src-tauri/src/lib.rs`：桌面端与 Python Worker 桥接；
- `video_page_detector/desktop_v2_worker.py`：任务队列和阶段流水线；
- `video_page_detector/pipeline.py`：PPT 页面检测；
- `video_page_detector/transcription.py`：语音转写；
- `video_page_detector/llm_evaluation.py`：关联度评分与报告。
