# 旧版 FFmpeg 场景阈值检测器

这里保存的是项目最初采用的检测方法，已经与当前主程序完全分离。

旧方法的流程：

1. 使用 FFmpeg `scene` 滤镜寻找画面突变候选点。
2. 在候选点前后进行分块 pHash 比较。
3. 在候选点后的短窗口内寻找稳定帧。
4. 使用“全局相似度 + 内容分块变化比例”联合去重。
5. 对过短页面进行合并。
6. 从原视频重新提取高清代表截图。

旧版最初只使用全局 pHash 去重，同模板的不同页面容易被误删。现在只有全局相似且内容变化分块少于 `duplicate_changed_block_ratio` 时才判为重复。该方法仍可能受到教师走动、渐变动画和曝光变化影响，因此保留独立 GUI 与参数。

关键优化参数：

- `duplicate_hash_distance`：全局哈希相似度初筛，默认 6。
- `duplicate_changed_block_ratio`：内容变化分块低于该比例才允许去重，默认 0.25。
- `auto_detect_screen_crop`：自动识别投影区域，默认开启。

## 单独运行

### 图形界面

双击项目根目录中的：

```text
启动旧版FFmpeg方法GUI.bat
```

或者执行：

```powershell
python -m legacy_ffmpeg_scene_detector --gui
```

界面把参数分成“常用检测参数”和“高级参数”两个页签。处理完成后会显示每页的起止时间、置信度和截图路径，双击结果行可打开截图。

### 命令行

在项目根目录执行：

```powershell
python -m legacy_ffmpeg_scene_detector "E:\leeson\3333.mp4"
```

指定配置和输出目录：

```powershell
python -m legacy_ffmpeg_scene_detector "E:\leeson\3333.mp4" `
  --config "E:\leeson\legacy_ffmpeg_scene_detector\config\default.json" `
  --output "E:\leeson\legacy_output"
```

运行旧版测试：

```powershell
python -m unittest discover -s legacy_ffmpeg_scene_detector/tests -v
```

目录中的 `pipeline.py` 是旧方法的主流程，`gui.py` 是独立界面，`ffmpeg_io.py` 包含 FFmpeg 场景阈值筛选，其他文件均为旧流程自己的配置、图像分析和后处理代码。
