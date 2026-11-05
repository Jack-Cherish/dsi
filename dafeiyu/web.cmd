@echo off
cd /d "%~dp0"
uv run --no-project --python 3.11 python scripts/run.py web
if errorlevel 1 pause
