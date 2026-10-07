# 旧版 FFmpeg 场景阈值检测器

这里保存的是项目最初采用的检测方法，现由桌面端的“场景阈值算法（FFmpeg）”选项调用。

旧方法的流程：

1. 使用 FFmpeg `scene` 滤镜寻找画面突变候选点。
2. 在候选点前后进行分块 pHash 比较。
3. 在候选点后的短窗口内寻找稳定帧。
4. 使用“全局相似度 + 内容分块变化比例”联合去重。
5. 对过短页面进行合并。
6. 从原视频重新提取高清代表截图。

旧版最初只使用全局 pHash 去重，同模板的不同页面容易被误删。现在只有全局相似且内容变化分块少于 `duplicate_changed_block_ratio` 时才判为重复。该方法仍可能受到教师走动、渐变动画和曝光变化影响。

关键优化参数：

- `duplicate_hash_distance`：全局哈希相似度初筛，默认 6。
- `duplicate_changed_block_ratio`：内容变化分块低于该比例才允许去重，默认 0.25。
- `auto_detect_screen_crop`：自动识别投影区域，默认开启。

## 测试

在 `场景阈值方法` 目录下运行旧版测试：

```powershell
python -m unittest discover -s legacy_ffmpeg_scene_detector/tests -v
```

目录中的 `pipeline.py` 是该算法的主流程，`ffmpeg_io.py` 包含 FFmpeg 场景阈值筛选，其他文件均为该算法自己的配置、图像分析和后处理代码。
