# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path
import shutil

from PyInstaller.utils.hooks import collect_all


project_root = Path(SPECPATH)
ffmpeg_path = shutil.which("ffmpeg")
ffprobe_path = shutil.which("ffprobe")
if not ffmpeg_path or not ffprobe_path:
    raise RuntimeError(
        "FFmpeg and FFprobe must be available on PATH before building."
    )

datas = [
    (str(project_root / "config"), "config"),
    (
        str(project_root / "models" / "faster-whisper"),
        "models/faster-whisper",
    ),
]
binaries = [
    (
        ffmpeg_path,
        "tools",
    ),
    (
        ffprobe_path,
        "tools",
    ),
]
hiddenimports = []

for package_name in (
    "faster_whisper",
    "ctranslate2",
    "tokenizers",
    "av",
    "huggingface_hub",
    "httpx",
    "httpcore",
    "certifi",
):
    package_datas, package_binaries, package_hiddenimports = collect_all(
        package_name
    )
    datas += package_datas
    binaries += package_binaries
    hiddenimports += package_hiddenimports

a = Analysis(
    [str(project_root / "desktop_launcher.py")],
    pathex=[str(project_root)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "matplotlib",
        "pandas",
        "scipy",
        "IPython",
        "jupyter",
        "notebook",
    ],
    noarchive=False,
    optimize=1,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="课堂PPT智能处理",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="课堂PPT智能处理",
)
