"""Train the defect model: python -m src.train

Dataset is small (500 rows, ~4% defects), so we use stratified repeated CV,
class weighting, PR-AUC for model selection and out-of-fold predictions to pick
the alert threshold.
"""
import json

import joblib
import warnings

import matplotlib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, classification_report,
                             confusion_matrix, precision_recall_curve, roc_auc_score)
from sklearn.model_selection import (RepeatedStratifiedKFold, StratifiedKFold,
                                     cross_val_predict, cross_validate,
                                     train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import (CATEGORICAL, FEATURES, NUMERIC, TARGET, load)

warnings.filterwarnings("ignore")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SEED = 42


def make_pipe(clf):
    pre = ColumnTransformer([("num", StandardScaler(), NUMERIC),
                             ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)])
    return Pipeline([("pre", pre), ("clf", clf)])


CANDIDATES = {
    "logistic_regression": LogisticRegression(class_weight="balanced", max_iter=2000, C=0.5),
    "random_forest": RandomForestClassifier(n_estimators=400, min_samples_leaf=2,
                                            class_weight="balanced_subsample",
                                            random_state=SEED, n_jobs=-1),
    "gradient_boosting": GradientBoostingClassifier(n_estimators=150, max_depth=2,
                                                    learning_rate=0.05, subsample=0.8,
                                                    random_state=SEED),
}


def main():
    df = load()
    X, y = df[FEATURES], df[TARGET]
    print(f"Rows={len(df)}  defects={y.sum()} ({y.mean():.1%})\n")

    # 1. Compare models with repeated stratified CV
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED)
    scores = {}
    for name, clf in CANDIDATES.items():
        r = cross_validate(make_pipe(clf), X, y, cv=cv, n_jobs=1,
                           scoring={"pr_auc": "average_precision", "roc_auc": "roc_auc",
                                    "recall": "recall", "precision": "precision"})
        scores[name] = {k[5:]: float(v.mean()) for k, v in r.items() if k.startswith("test_")}
        print(name, {k: round(v, 3) for k, v in scores[name].items()})
    best = max(scores, key=lambda n: scores[n]["pr_auc"])
    print(f"\nSelected model: {best}\n")

    # 2. Out-of-fold probabilities -> alert threshold (recall >= 75%, best precision)
    oof = cross_val_predict(make_pipe(CANDIDATES[best]), X, y,
                            cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
                            method="predict_proba")[:, 1]
    prec, rec, thr = precision_recall_curve(y, oof)
    ok = [(p, t) for p, r, t in zip(prec[:-1], rec[:-1], thr) if r >= 0.75]
    threshold = float(max(ok)[1]) if ok else 0.5
    pred = (oof >= threshold).astype(int)
    print(f"Alert threshold={threshold:.3f}")
    print(classification_report(y, pred, target_names=["ok", "defect"]))
    print("Confusion matrix:\n", confusion_matrix(y, pred))

    # 3. Hold-out sanity check + final fit on all data
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=SEED)
    hold = make_pipe(CANDIDATES[best]).fit(Xtr, ytr).predict_proba(Xte)[:, 1]
    holdout = {"roc_auc": float(roc_auc_score(yte, hold)),
               "pr_auc": float(average_precision_score(yte, hold))}
    print("Hold-out:", holdout)

    model = make_pipe(CANDIDATES[best]).fit(X, y)
    imp = permutation_importance(model, X, y, scoring="average_precision",
                                 n_repeats=20, random_state=SEED)
    importance = dict(sorted(zip(FEATURES, imp.importances_mean.clip(min=0).tolist()),
                             key=lambda kv: -kv[1]))
    print("Feature importance:", {k: round(v, 3) for k, v in importance.items()})

    # Nominal operating point = healthy samples
    good = df[y == 0]
    nominal = {c: float(good[c].median()) for c in NUMERIC}
    spread = {c: float(good[c].std()) for c in NUMERIC}

    joblib.dump({"model": model, "threshold": threshold, "nominal": nominal,
                 "spread": spread, "importance": importance,
                 "machines": sorted(df["machine_id"].unique().tolist())},
                "models/defect_model.joblib")
    with open("models/metrics.json", "w") as f:
        json.dump({"selected": best, "cv_scores": scores, "holdout": holdout,
                   "alert_threshold": threshold, "importance": importance}, f, indent=2)

    # 4. Plots
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(rec, prec); ax[0].set(title="Precision-Recall (out-of-fold)",
                                     xlabel="Recall", ylabel="Precision")
    ax[1].barh(list(importance)[::-1], list(importance.values())[::-1])
    ax[1].set(title="Permutation importance (PR-AUC drop)")
    fig.tight_layout(); fig.savefig("models/evaluation.png", dpi=120)
    print("Saved models/defect_model.joblib, metrics.json, evaluation.png")


if __name__ == "__main__":
    main()
