@echo off
rem Saqta TZ Auditor: one-command check of the case document.
rem Usage: start_report.bat [input.pdf] [output.md]
cd /d "%~dp0"
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
set IN=%~1
set OUT=%~2
if "%IN%"=="" set IN=examples\case_requirements.pdf
if "%OUT%"=="" set OUT=reports\report.md
python validate.py "%IN%" "%OUT%"
echo.
echo Report: %OUT%
start "" "%OUT%"
pause
