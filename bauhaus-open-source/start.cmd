@echo off
cd /d "%~dp0"
where node >nul 2>nul
if errorlevel 1 (
  echo Node.js is not installed. Open standalone.html instead.
  start "" "%~dp0standalone.html"
  exit /b
)
start "" "http://127.0.0.1:18769/"
node scripts/serve.mjs
pause
