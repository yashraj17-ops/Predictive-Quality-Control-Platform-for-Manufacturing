# Predictive Quality Control Platform for Manufacturing

ML model + FastAPI backend + web frontend. Predicts defect probability from machine and
process parameters, ranks the key causes, and raises alerts with corrective actions.

```
pip install -r requirements.txt
python -m src.train                # trains on small_custom_manufacturing_dataset_500.csv
uvicorn backend:app --reload       # open http://127.0.0.1:8000
```

- `src/data.py`, `src/train.py`, `src/predict.py` – data, training/evaluation (CV, threshold), inference + causes
- `backend.py` – REST API: `/api/predict`, `/api/batch`, `/api/sample`, `/api/alerts`, `/api/config`, `/api/metrics` (docs at `/docs`)
- `frontend/` – Live Check, Batch Monitor (CSV upload) and Alerts pages
- `app.py` – optional older Streamlit UI (needs `pip install streamlit`)

Note: only 20 defects in 500 rows, so expect modest precision; more defect data will help most.
