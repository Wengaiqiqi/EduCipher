@echo off
chcp 65001 >nul
cd /d "%~dp0"
python -m PyInstaller --noconfirm build_desktop_v2_worker.spec
if errorlevel 1 goto :failed
cd /d "%~dp0desktop_v2"
set "PATH=%USERPROFILE%\.cargo\bin;%PATH%"
call C:\BuildTools\Common7\Tools\VsDevCmd.bat -arch=x64 -host_arch=x64 >nul
call npm run dist:win
if errorlevel 1 goto :failed
echo.
echo 构建完成，请查看 desktop_v2\src-tauri\target\release\bundle\nsis
pause
exit /b 0

:failed
echo.
echo 构建失败，请查看上方错误信息。
pause
exit /b 1
