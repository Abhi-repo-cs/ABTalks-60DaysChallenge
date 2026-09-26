# Day 55 — Customer Intelligence Platform: Scalability & Reliability

A production-oriented Day 55 capstone focused on performance, reliability, validation, and scalability.

## Run frontend
```powershell
npm install
npm run dev
```

## Run API
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install fastapi uvicorn pydantic
uvicorn api.main:app --reload
```

Open `http://127.0.0.1:8000/docs`.

## Improvements
- Memoized dashboard calculations
- Reduced repeated computations
- Input validation
- Structured error handling
- Health/readiness endpoints
- Latency instrumentation
- Scalability architecture
- Reliability test checklist

This is a learning/demo implementation. Production performance must be benchmarked on representative infrastructure and workloads.
