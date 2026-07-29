@echo off
set "INSTALL_SCRIPT=%~dp0install_local_ffmpeg_admin.ps1"
powershell.exe -NoProfile -ExecutionPolicy Bypass -Command ^
  "$process = Start-Process -FilePath 'powershell.exe' -Verb RunAs -Wait -PassThru -ArgumentList @('-NoProfile','-ExecutionPolicy','Bypass','-File','%INSTALL_SCRIPT%'); exit $process.ExitCode"
if errorlevel 1 (
  echo.
  echo FFmpeg installation did not complete successfully.
) else (
  echo.
  echo FFmpeg installation completed. Reopen terminals and the GUI.
)
pause
