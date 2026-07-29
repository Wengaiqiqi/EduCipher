@echo off
chcp 65001 >nul
cd /d "%~dp0"
python -m video_page_detector transcribe-gui
if errorlevel 1 pause
