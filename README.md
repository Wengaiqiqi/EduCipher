# 课堂视频 PPT 自动提取器

主程序只保留当前的“时间连续性检测”方法：自动识别投影区域，持续比较画面状态，确认稳定换页后选择当前页面中信息量最大的代表帧，并输出页面起止时间。

旧的 FFmpeg 场景阈值方法已移至
[`场景阈值方法/legacy_ffmpeg_scene_detector`](场景阈值方法/legacy_ffmpeg_scene_detector/README.md)，
不会再被当前 GUI 或主程序加载。需要使用旧方法时，可双击
`场景阈值方法/启动旧版FFmpeg方法GUI.bat` 打开它自己的独立界面。

## 推荐：直接使用 Windows 桌面应用

发布版位于：

```text
dist\课堂PPT智能处理\课堂PPT智能处理.exe
```

双击 EXE 后即可在同一个中文界面完成：

1. 从视频自动提取 PPT 页面及时间区间；
2. 使用内置 `small` 模型或小米 MiMo 云端把讲话转成逐页文字；
3. 使用 OpenAI 兼容多模态 LLM 逐页评分；
4. 查看页数、讲话段数、逐页分数并打开最终报告。

发布版采用 Windows 单目录形式，已经包含 Python 运行环境、FFmpeg、
FFprobe 和本地语音模型，约 1GB。移动或复制时必须保留整个
`课堂PPT智能处理` 文件夹，不能只复制其中的 EXE。

界面提供三个页签：

- `一键处理`：选择视频后运行完整流程，也可只提取 PPT；
- `继续已有结果`：从 `result.json` 继续转写，或从
  `transcript.json` 继续评分；
- `模型与设置`：选择本地或小米云端语音识别，填写专业词汇、
  云端并发数、LLM地址、模型和API Key。

API Key只保留在本次运行内存中，任务结束后自动清空。发送课堂音频
给小米MiMo、发送截图和转写给第三方LLM都分别需要主动勾选上传确认。

默认结果保存在：

```text
文档\课堂PPT处理结果\<任务名称>\
```

开发环境也可运行统一界面：

```powershell
python -m video_page_detector desktop-gui
```

### 从GitHub源码构建EXE

Git仓库不包含测试视频、运行输出、约526MB的本地语音模型和约1GB的
`dist` 发布目录，以避免超过GitHub普通仓库的单文件限制。

构建前需要：

1. 安装Python 3.10或更高版本及项目依赖；
2. 确保 `ffmpeg` 和 `ffprobe` 已加入PATH；
3. 运行一次语音转写，让 `small` 模型下载到
   `models/faster-whisper`；
4. 安装PyInstaller，然后双击 `构建Windows应用.bat`。

构建结果会生成在：

```text
dist\课堂PPT智能处理\课堂PPT智能处理.exe
```

## 当前处理流程

1. 每 2 秒提取一张低分辨率分析帧。
2. 自动定位投影区域，减少教师、墙面和黑边干扰。
3. 对投影区域进行分块感知哈希比较。
4. 新状态持续约 10 秒后确认换页。
5. 比较位置无关的文字行内容，合并缩放、行距变化和重排等同页动画。
6. 以 0.25 秒精度回查并细化换页时间。
7. 在同一页面中选择信息增量最大的代表帧。
8. 从原视频重新提取全分辨率截图，并写入页面起止时间。

分析帧会缩小以保证速度，但最终截图使用视频源分辨率。开启投影区域裁剪后，图片尺寸会小于 1920×1080，但不会降采样成低清图片。

## 启动 GUI

双击：

```text
启动GUI.bat
```

或者执行：

```powershell
python -m video_page_detector gui
```

在 GUI 中选择视频和输出目录，然后点击开始处理。处理完成后，双击结果行可以打开对应截图。

## 将视频讲话按 PPT 页面转成文字

PPT 检测完成后，双击：

```text
启动语音转文字GUI.bat
```

在界面中选择：

1. 原课堂视频；
2. 新算法或旧算法生成的 `result.json`；
3. 文字输出目录；
4. 本地识别模型和课程专业词汇，或选择小米MiMo云端模式。

第一版保留视频中所有能够识别的讲话，不区分老师和学生，不清理
“嗯、啊、然后”等口头语。程序直接读取视频音轨，不生成或保留
WAV、MP3 等音频文件。

默认使用 CPU `int8` 的 `small` 模型。模型首次使用时会下载到
`models/faster-whisper`，以后无需重复下载。专业词汇可以在 GUI
中按课程修改，以提高术语识别率。

### 小米MiMo云端语音识别

在“识别方式”中选择 `小米 MiMo 云端（推荐加速）` 后，程序使用
小米官方模型 `mimo-v2.5-asr` 和
`https://api.xiaomimimo.com/v1/chat/completions`。

云端模式会：

1. 按现有PPT开始、结束时间提取16kHz单声道临时WAV；
2. 默认同时处理3页；
3. 每页独立调用小米ASR，因此不依赖服务端时间戳；
4. 请求结束或发生错误后立即删除临时音频；
5. 保持原有 `transcript.json` 结构，后续LLM评分无需改变。

API Key可在界面中临时输入，也可通过用户环境变量配置：

```powershell
setx MIMO_API_KEY "你的密钥"
```

为兼容现有配置，未找到 `MIMO_API_KEY` 时也会尝试
`LLM_API_KEY`。密钥不会写入结果、设置或日志。使用云端前必须勾选
“允许把临时音频发送给小米MiMo”。

命令行方式：

```powershell
python -m video_page_detector transcribe `
  "E:\leeson\test_vedio\29_16.mp4" `
  "E:\leeson\output\29_16\result.json" `
  --config "E:\leeson\config\transcription.json" `
  --engine mimo-cloud
```

命令行方式：

```powershell
python -m video_page_detector transcribe `
  "E:\leeson\test_vedio\29_16.mp4" `
  "E:\leeson\output\29_16\result.json" `
  --config "E:\leeson\config\transcription.json"
```

输出文件：

```text
transcript.json
逐页语音文字.md
```

一句话跨越换页时不会被截断，而是按讲话时间占比归入对应页面，
并保留原始开始和结束时间。

程序会在加载模型之前校验视频时长与 PPT 时间轴。测试片段、截短视频
或选错 `result.json` 时会直接报错，不会生成后半部分全为空的误导结果。

## 使用 LLM 评估 PPT 与讲话关联度

逐页语音文字生成后，双击：

```text
启动LLM关联度评估GUI.bat
```

输入支持图片的 OpenAI 兼容多模态服务：

- `Base URL`，例如 `https://服务地址/v1`；
- 模型名称；
- API密钥；
- 并发数量，默认5，允许1～10。

API密钥只保存在当前GUI内存中，任务结束后清空，不会写入配置、结果或日志。
命令行模式从环境变量 `LLM_API_KEY` 读取密钥。

每一页独立发送当前PPT截图和已按页面时间区间归类的纯讲话文字。
逐句时间戳只保留在本地 `transcript.json` 中，不发送给LLM，页面分数由程序统一计算：

```text
页面分数 =
讲话相关度 × 60%
+ PPT覆盖度 × 25%
+ 证据一致性 × 15%
```

最终同时输出：

- 严格总分：所有页面得分之和 ÷ PPT总页数；
- 纯关联平均分：有讲话页面得分之和 ÷ 有讲话页面数；
- 讲话页面覆盖率；
- 每页关键点、对应证据、分项得分、总分和理由。

无讲话页面按0分计入严格总分，但不会请求LLM。请求失败的页面不会被当成
0分；任务会标记为未完成，重新运行时仅补跑失败或输入发生变化的页面。

输出结构：

```text
llm_evaluation/
├── pages/
│   ├── page_001.json
│   └── page_NNN.json
├── llm_evaluation.json
└── PPT讲话关联度报告.md
```

命令行方式：

```powershell
$env:LLM_API_KEY = "本次运行使用的密钥"
python -m video_page_detector evaluate-llm `
  "E:\leeson\diagnostics\intro_merge\29_16\transcript.json" `
  --config "E:\leeson\config\llm_evaluation.json"
```

## 命令行运行

```powershell
python -m video_page_detector detect "E:\leeson\3333.mp4" `
  --config "E:\leeson\config\default.json" `
  --output "E:\leeson\output"
```

## 当前配置

| 参数 | 默认值 | 作用 |
|---|---:|---|
| `temporal_sample_interval_sec` | 2.0 | 时序粗采样间隔 |
| `temporal_confirmation_sec` | 10.0 | 新状态持续多久才确认换页 |
| `temporal_changed_block_ratio` | 0.5 | 判定页面变化的分块比例 |
| `temporal_same_content_similarity` | 0.8 | 合并文字内容相同、仅位置或行距变化的动画状态 |
| `temporal_analysis_width` | 320 | 分析帧宽度 |
| `temporal_analysis_height` | 180 | 分析帧高度 |
| `auto_detect_screen_crop` | true | 自动识别投影区域 |
| `screen_crop_*_ratio` | 见配置 | 自动识别失败时的裁剪后备值 |
| `crop_output_screenshots` | true | 最终截图只保留投影区域 |
| `temporal_refinement_fps` | 4.0 | 换页时间精修帧率 |
| `min_page_duration_sec` | 5.0 | 页面最短持续时间 |
| `jpeg_quality` | 90 | 最终截图 JPEG 质量 |

默认配置文件是 [`config/default.json`](config/default.json)。

## 输出

每个视频会生成：

```text
output/<video_id>/
├── page_001.jpg
├── page_002.jpg
├── result.json
└── temporal/
    └── segments.json
```

`result.json` 包含每页的 `start_sec`、`end_sec`、`representative_sec`、截图路径、源视频分辨率和实际截图分辨率。

## 测试

当前主程序（含统一桌面应用）：

```powershell
python -m unittest discover -s tests -v
```

归档旧方法：

```powershell
python -m unittest discover -s legacy_ffmpeg_scene_detector/tests -v
```
