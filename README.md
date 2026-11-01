# Predictive Quality Control Platform for Manufacturing

ML model + FastAPI backend + web frontend. Predicts defect probability from machine and
process parameters, ranks the key causes, and raises alerts with corrective actions.

```
pip install -r requirements.txt
python -m src.train                # trains on small_custom_manufacturing_dataset_500.csv
uvicorn main:app --reload       # open http://127.0.0.1:8000
```

- `src/data.py`, `src/train.py`, `src/predict.py` – data, training/evaluation (CV, threshold), inference + causes
- `main.py` – REST API: `/api/predict`, `/api/batch`, `/api/sample`, `/api/alerts`, `/api/config`, `/api/metrics` (docs at `/docs`)
- `frontend/` – Live Check, Batch Monitor (CSV upload) and Alerts pages

Note: only 20 defects in 500 rows, so expect modest precision; more defect data will help most.

## Deploy on Vercel
Import the repo, choose the **FastAPI** preset (no build command). `vercel.json` bundles the
frontend, model and dataset. If the saved model can't load on Vercel's Python/scikit-learn
version it is refit from the CSV on cold start. Training needs `pip install -r requirements-train.txt`.
The in-memory Alerts log is per serverless instance and may reset.
