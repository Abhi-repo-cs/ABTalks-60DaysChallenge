from time import perf_counter
from fastapi import FastAPI,HTTPException
from pydantic import BaseModel,Field
app=FastAPI(title="Customer Intelligence Reliability API",version="1.0.0")
class PredictionRequest(BaseModel):
 age:int=Field(ge=18,le=100); income:float=Field(gt=0); tenure_months:int=Field(ge=0); orders:int=Field(ge=0); total_spend:float=Field(ge=0); complaints:int=Field(ge=0)
@app.get("/health")
def health(): return {"status":"healthy","service":"customer-intelligence"}
@app.get("/ready")
def ready(): return {"status":"ready","model_loaded":True}
@app.post("/predict")
def predict(p:PredictionRequest):
 start=perf_counter()
 try:
  score=.15+(0.30 if p.orders<=3 else 0)+(0.20 if p.complaints>=2 else 0)+(0.15 if p.tenure_months<18 else 0)+(0.10 if p.total_spend<20000 else 0)
  score=min(score,.99)
  return {"churn_probability":round(score,4),"risk_band":"High" if score>=.6 else "Medium" if score>=.35 else "Low","latency_ms":round((perf_counter()-start)*1000,3)}
 except Exception as e: raise HTTPException(status_code=500,detail="Prediction service failed safely") from e
