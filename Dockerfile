FROM python:3.10-slim

WORKDIR /app

# system deps for pandas/openpyxl
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

# default command runs the orchestrator on the sample CSV
CMD ["python", "run_orchestrator.py", "--data", "data/sample_input.csv"]
