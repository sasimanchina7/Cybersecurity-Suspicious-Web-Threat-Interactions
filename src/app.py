from fastapi import FastAPI
import mlflow.sklearn
import numpy as np
from pydantic import BaseModel

app = FastAPI(title="ThreatDetectorAPI")
model = mlflow.sklearn.load_model("models:/ThreatModel/Production")
preproc = mlflow.sklearn.load_model("models:/Preprocessor/Production")  # if logged

class Payload(BaseModel):
    bytes_in: int
    bytes_out: int
    protocol: str
    src_ip_country_code: str
    creation_time: str
    end_time: str

@app.post("/predict")
def predict(payload: Payload):
    import pandas as pd
    df = pd.DataFrame([payload.dict()])
    X = preproc.transform(df)
    prob = model.predict_proba(X)[0,1]
    return {"suspicious_score": float(prob)}
