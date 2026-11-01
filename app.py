import json

import pandas as pd
import streamlit as st

from src.data import FEATURES, NUMERIC, UNITS, load
from src.predict import bundle, predict, predict_batch

st.set_page_config(page_title="Predictive Quality Control", layout="wide")
st.title("Predictive Quality Control Platform")

b = bundle()
ref = load()
tab1, tab2 = st.tabs(["Live Check", "Batch Monitor"])

with tab1:
    machine = st.selectbox("Machine", b["machines"])
    cols = st.columns(3)
    reading = {"machine_id": machine}
    for i, c in enumerate(NUMERIC):
        lo, hi = float(ref[c].min()), float(ref[c].max())
        reading[c] = cols[i % 3].slider(f"{c} ({UNITS[c]})", lo, hi, b["nominal"][c])
    r = predict(reading)
    left, right = st.columns([1, 2])
    left.metric("Defect probability", f"{r['probability']:.1%}")
    {"HIGH": left.error, "MEDIUM": left.warning, "LOW": left.success}[r["level"]](
        f"Risk level: {r['level']}")
    for a in r["actions"]:
        left.markdown(f"**Action:** {a}")
    right.subheader("Key causes (risk removed if reset to normal)")
    right.bar_chart(pd.DataFrame(r["causes"]).set_index("feature")["risk_contribution"])

with tab2:
    up = st.file_uploader("CSV with columns: " + ", ".join(FEATURES), type="csv")
    df = pd.read_csv(up) if up else ref.head(200)
    res = predict_batch(df)
    st.write(f"{(res.risk == 'HIGH').sum()} HIGH-risk of {len(res)} readings")
    st.dataframe(res.sort_values("defect_prob", ascending=False), use_container_width=True)

with st.sidebar:
    st.header("Model")
    with open("models/metrics.json") as f:
        st.json(json.load(f))
