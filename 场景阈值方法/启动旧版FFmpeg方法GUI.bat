@echo off
chcp 65001 >nul
cd /d "%~dp0"
python -m legacy_ffmpeg_scene_detector --gui
if errorlevel 1 pause
