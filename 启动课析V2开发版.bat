@echo off
chcp 65001 >nul
cd /d "%~dp0desktop_v2"
set "PATH=%USERPROFILE%\.cargo\bin;%PATH%"
call C:\BuildTools\Common7\Tools\VsDevCmd.bat -arch=x64 -host_arch=x64 >nul
call npm run dev
pause
