@echo off
cd /d "%~dp0"
python -m video_page_detector gui
if errorlevel 1 (
  echo.
  echo GUI failed to start. Please check Python and project dependencies.
  pause
)
