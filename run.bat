@echo off
setlocal

cd /d "%~dp0"

set "PY_CMD="
where py >nul 2>nul
if %errorlevel%==0 (
    set "PY_CMD=py"
) else (
    where python >nul 2>nul
    if %errorlevel%==0 (
        set "PY_CMD=python"
    )
)

if "%PY_CMD%"=="" (
    echo [ERROR] Python launcher not found. Install Python and ensure py or python is on PATH.
    exit /b 1
)

if /I "%~1"=="test" (
    echo Running test suite...
    %PY_CMD% -m pytest tests/ -v
    exit /b %errorlevel%
)

if /I "%~1"=="help" goto :usage
if /I "%~1"=="-h" goto :usage
if /I "%~1"=="--help" goto :usage
if /I "%~1"=="/?" goto :usage

if not exist "main.py" (
    echo [ERROR] main.py not found. Run this script from the project root.
    exit /b 1
)

if not exist "stable-diffusion-v1-5" (
    echo [WARN] stable-diffusion-v1-5 folder not found.
)
if not exist "sd-controlnet-scribble" (
    echo [WARN] sd-controlnet-scribble folder not found.
)

echo Starting cv_scribble_diffusion...
%PY_CMD% main.py
exit /b %errorlevel%

:usage
echo Usage:
echo   run.bat         ^(start app^)
echo   run.bat test    ^(run tests^)
echo   run.bat help    ^(show this message^)
exit /b 0
