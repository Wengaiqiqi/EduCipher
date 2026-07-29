@echo off
chcp 65001 >nul
cd /d "%~dp0"

python -m PyInstaller --noconfirm --clean build_desktop_app.spec
if errorlevel 1 (
  echo.
  echo 构建失败，请查看上面的错误信息。
  pause
  exit /b 1
)

copy /y "%~dp0桌面应用使用说明.txt" "%~dp0dist\课堂PPT智能处理\使用说明.txt" >nul

echo.
echo 构建完成：
echo %~dp0dist\课堂PPT智能处理\课堂PPT智能处理.exe
pause
