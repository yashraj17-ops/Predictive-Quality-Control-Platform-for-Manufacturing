"""Inference + per-reading root-cause attribution (counterfactual, model-based)."""
import json

import joblib
import pandas as pd

from .data import FEATURES, NUMERIC, UNITS

_bundle = None


def _rebuild():
    """Refit the selected model from the CSV (used if the pickle can't be loaded,
    e.g. scikit-learn version mismatch on the deployment platform)."""
    from .data import TARGET, load
    from .train import CANDIDATES, make_pipe
    with open("models/metrics.json") as f:
        m = json.load(f)
    df = load()
    model = make_pipe(CANDIDATES[m["selected"]]).fit(df[FEATURES], df[TARGET])
    good = df[df[TARGET] == 0]
    return {"model": model, "threshold": m["alert_threshold"],
            "nominal": {c: float(good[c].median()) for c in NUMERIC},
            "spread": {c: float(good[c].std()) for c in NUMERIC},
            "importance": m["importance"],
            "machines": sorted(df["machine_id"].unique().tolist())}


def bundle():
    global _bundle
    if _bundle is None:
        try:
            _bundle = joblib.load("models/defect_model.joblib")
        except Exception:
            _bundle = _rebuild()
    return _bundle


def risk_level(p, thr):
    return "HIGH" if p >= thr else "MEDIUM" if p >= thr / 2 else "LOW"


def predict(reading: dict) -> dict:
    b = bundle()
    model, nominal, spread = b["model"], b["nominal"], b["spread"]
    x = pd.DataFrame([reading])[FEATURES]
    p = float(model.predict_proba(x)[0, 1])

    # Risk removed if a parameter is reset to its healthy (median) value
    causes = []
    for c in NUMERIC:
        x2 = x.copy()
        x2[c] = nominal[c]
        drop = p - float(model.predict_proba(x2)[0, 1])
        causes.append({"feature": c, "value": reading[c], "nominal": nominal[c],
                       "unit": UNITS[c], "z": (reading[c] - nominal[c]) / spread[c],
                       "risk_contribution": drop})
    causes.sort(key=lambda d: d["risk_contribution"], reverse=True)

    level = risk_level(p, b["threshold"])
    actions = []
    if level != "LOW":
        for d in causes[:2]:
            if d["risk_contribution"] > 0.01:
                if d["feature"] == "tool_wear_min":
                    actions.append("Schedule tool change / inspection")
                else:
                    verb = "Reduce" if d["z"] > 0 else "Increase"
                    actions.append(f"{verb} {d['feature']} toward {d['nominal']:.1f} {d['unit']}")
    return {"probability": p, "level": level, "threshold": b["threshold"],
            "causes": causes, "actions": actions}


def predict_batch(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    res = [predict(r) for r in df[FEATURES].to_dict("records")]
    out["defect_prob"] = [r["probability"] for r in res]
    out["risk"] = [r["level"] for r in res]
    out["top_cause"] = [r["causes"][0]["feature"] for r in res]
    return out
