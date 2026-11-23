FROM python:3.12-slim

WORKDIR /app
ENV PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

EXPOSE 8501
# Ключ LLM передаётся переменными окружения: docker run -e LLM_API_KEY=... -e LLM_BASE_URL=... -p 8501:8501 tz-auditor
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true", "--browser.gatherUsageStats=false"]
