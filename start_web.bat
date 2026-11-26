@echo off
rem Saqta TZ Auditor: web interface. Opens http://localhost:8501 in the browser.
cd /d "%~dp0"
python -m streamlit run app.py --server.port 8501
pause
