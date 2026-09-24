# Day 53 — Customer Intelligence Real-Time Analytics

React + Vite prototype for live customer churn-risk prediction.

## Run — Windows PowerShell
```powershell
cd "C:\path\to\Day53_Customer_Intelligence_Real_Time_Analytics"
npm install
npm run dev
```
Open the localhost URL shown by Vite, normally `http://localhost:5173`.

## Production build
```powershell
npm run build
npm run preview
```

## Features
- Real-time prediction form
- Live customer input submission
- Instant churn-risk output
- Probability and confidence
- Decision signals
- Dashboard KPI integration
- Prediction event stream
- Workflow documentation
- Responsive UI

## Production integration
Replace the `predict()` function in `src/main.jsx` with a request to a FastAPI/Flask model-serving endpoint backed by the optimized model.

## Submission
- [x] Real-time analytics application
- [ ] Prediction demo screenshots
- [x] Workflow explanation report
- [x] GitHub-ready project
- [x] LinkedIn reflection
