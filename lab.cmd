@echo off
rem Start the lab on Windows: lab   (Ctrl+C stops it; run lab again to restart)
rem   lab mcp    runs the MCP server an agent (Claude Code, Codex, ...) connects to
rem Same as ./lab on Mac and Linux.
cd /d "%~dp0"

rem A clone is not an install: install.ps1 builds .venv, and cloning with git skips it.
rem Without this the launcher printed a raw "cannot find the path" from cmd, which names
rem nothing the reader can act on.
if not exist ".venv\Scripts\python.exe" (
  echo The lab's Python environment is missing (.venv\Scripts\python.exe).
  echo.
  echo If you cloned this repo instead of running the installer, set it up with:
  echo.
  echo     uv venv --python 3.11 .venv
  echo     uv pip install --python .venv\Scripts\python.exe -r requirements.lock
  echo.
  echo or run the installer, which does that and the machine checks for you:
  echo.
  echo     irm https://assetfurnace.com/install.ps1 ^| iex
  exit /b 1
)

if /i "%~1"=="mcp" (
  ".venv\Scripts\python.exe" -m image_to_3dlab.mcp_server %2 %3 %4 %5 %6 %7 %8 %9
  exit /b
)
".venv\Scripts\python.exe" viewer\serve.py --auto %*
