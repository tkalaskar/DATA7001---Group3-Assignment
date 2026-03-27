FROM python:3.11-slim

LABEL maintainer="APEX Team — UQ DATA7001"
LABEL description="APEX: Autonomous Paradigm-fusing Explanation Engine"

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    graphviz \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV APEX_API_HOST=0.0.0.0
ENV APEX_API_PORT=8000

EXPOSE 8000

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
