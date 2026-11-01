"""FastAPI backend + static frontend.  Run: uvicorn backend:app --reload"""
import io
import json
from collections import deque
from datetime import datetime

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.data import FEATURES, NUMERIC, UNITS, load
from src.predict import bundle, predict, predict_batch

app = FastAPI(title="Predictive Quality Control API")
ALERTS = deque(maxlen=100)  # in-memory alert log of HIGH-risk predictions


class Reading(BaseModel):
    machine_id: str
    temperature_C: float
    pressure_bar: float
    vibration_mm_s: float
    rotational_speed_rpm: float
    production_speed_units_hr: float
    tool_wear_min: float


@app.get("/api/config")
def config():
    b, ref = bundle(), load()
    return {"machines": b["machines"], "nominal": b["nominal"], "units": UNITS,
            "ranges": {c: [float(ref[c].min()), float(ref[c].max())] for c in NUMERIC},
            "threshold": b["threshold"]}


@app.get("/api/metrics")
def metrics():
    with open("models/metrics.json") as f:
        return json.load(f)


@app.post("/api/predict")
def predict_one(r: Reading, log: bool = False):
    res = predict(r.model_dump())
    if log and res["level"] == "HIGH":
        ALERTS.appendleft({"time": datetime.now().strftime("%H:%M:%S"),
                           "machine": r.machine_id, "probability": res["probability"],
                           "top_cause": res["causes"][0]["feature"],
                           "actions": res["actions"]})
    return res


@app.post("/api/batch")
async def batch(file: UploadFile = File(...)):
    df = pd.read_csv(io.BytesIO(await file.read()))
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise HTTPException(400, f"Missing columns: {missing}")
    return _batch_json(df)


@app.get("/api/sample")
def sample():
    return _batch_json(load().head(200))


def _batch_json(df):
    out = predict_batch(df)
    cols = [c for c in ["sample_id"] + FEATURES + ["defect_prob", "risk", "top_cause"]
            if c in out.columns]
    out = out[cols].sort_values("defect_prob", ascending=False)
    return {"summary": out["risk"].value_counts().to_dict(), "total": len(out),
            "rows": json.loads(out.head(300).round(3).to_json(orient="records"))}


@app.get("/api/alerts")
def alerts():
    return list(ALERTS)


@app.get("/")
def index():
    return FileResponse("frontend/index.html")


app.mount("/static", StaticFiles(directory="frontend"), name="static")
