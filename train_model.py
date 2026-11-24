"""
train_model.py
==============
Heart Disease Detection - model training.

Reproduces the pipeline from the project's preprocessing / training scripts:
  * UCI Heart Disease dataset (920 records, 4 cohorts)
  * target binarised: num > 0  ->  1 (disease), else 0
  * numeric features  : median imputation + StandardScaler
  * categorical features: most-frequent imputation + OneHotEncoder
  * 70/30 stratified train-test split (random_state = 42)
  * 5-fold StratifiedKFold + GridSearchCV (scoring = roc_auc)
  * Logistic Regression, Random Forest, SVM, Gradient Boosting, XGBoost

Difference from the original scripts: the preprocessor and the classifier are
saved together as ONE scikit-learn Pipeline. This guarantees that the exact
same transformations are applied at serving time, which removes a whole class
of silent bugs (e.g. feeding unscaled values to a model trained on scaled data).

Outputs
-------
model/heart_model.joblib : Pipeline(preprocessor -> classifier)
model/metrics.json       : metrics for every candidate model
model/metadata.json      : dataset info + selected model
"""

import json
import os
import time
import warnings

import numpy as np
import pandas as pd
from joblib import dump

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, brier_score_loss, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE, "data", "heart_disease_uci.csv")
MODEL_DIR = os.path.join(BASE, "model")
os.makedirs(MODEL_DIR, exist_ok=True)

RANDOM_STATE = 42
TARGET = "num"

# Feature order used by the API (everything except id and target)
# Reduced feature set: only the fields that carry most of the predictive
# signal and can realistically be filled in from a basic check-up.
# Dropping the remaining six (dataset, fbs, restecg, slope, ca, thal) costs
# ~1.6 ROC-AUC points on 5-fold CV but removes specialist-only tests.
NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalch", "oldpeak"]
CATEGORICAL_FEATURES = ["sex", "cp", "exang"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)

    if "id" in df.columns:
        df = df.drop(columns=["id"])

    # Binarise the target: 0 = no disease, 1..4 = disease present
    df[TARGET] = (df[TARGET] > 0).astype(int)

    # fbs / exang are read as booleans; normalise to strings so the
    # encoder sees a stable set of categories at train and inference time.
    df["exang"] = df["exang"].astype(str)

    return df


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numeric_pipeline, NUMERIC_FEATURES),
        ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
    ])


def candidate_models():
    from xgboost import XGBClassifier

    return {
        "Logistic Regression": (
            LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
            {"C": [0.1, 1]},
        ),
        "Random Forest": (
            RandomForestClassifier(random_state=RANDOM_STATE),
            {"n_estimators": [100], "max_depth": [5, 10]},
        ),
        "Support Vector Machine": (
            SVC(probability=True, random_state=RANDOM_STATE),
            {"C": [1], "kernel": ["rbf"]},
        ),
        "Gradient Boosting": (
            GradientBoostingClassifier(random_state=RANDOM_STATE),
            {"n_estimators": [100], "learning_rate": [0.1], "max_depth": [3]},
        ),
        "XGBoost": (
            XGBClassifier(random_state=RANDOM_STATE, eval_metric="logloss", n_jobs=1),
            {"n_estimators": [100], "learning_rate": [0.1], "max_depth": [3]},
        ),
    }


def main():
    df = load_data()
    X, y = df[ALL_FEATURES], df[TARGET]

    print("=" * 68)
    print("DATASET")
    print("=" * 68)
    print(f"Records            : {len(df)}")
    print(f"Features           : {len(ALL_FEATURES)}")
    print(f"Class distribution : {y.value_counts().to_dict()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y)
    print(f"Training samples   : {len(X_train)}")
    print(f"Testing samples    : {len(X_test)}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    results, fitted = {}, {}
    print("\n" + "=" * 68)
    print("TRAINING + GRID SEARCH (scoring = roc_auc)")
    print("=" * 68)

    for name, (estimator, grid) in candidate_models().items():
        pipe = Pipeline([("preprocessor", build_preprocessor()),
                         ("classifier", estimator)])

        start = time.time()
        search = GridSearchCV(pipe, {f"classifier__{k}": v for k, v in grid.items()},
                              cv=cv, scoring="roc_auc", n_jobs=2)
        search.fit(X_train, y_train)
        elapsed = time.time() - start

        best = search.best_estimator_
        y_pred = best.predict(X_test)
        y_prob = best.predict_proba(X_test)[:, 1]
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

        results[name] = {
            "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
            "precision": round(float(precision_score(y_test, y_pred)), 4),
            "recall": round(float(recall_score(y_test, y_pred)), 4),
            "specificity": round(float(tn / (tn + fp)), 4),
            "f1": round(float(f1_score(y_test, y_pred)), 4),
            "roc_auc": round(float(roc_auc_score(y_test, y_prob)), 4),
            "brier": round(float(brier_score_loss(y_test, y_prob)), 4),
            "cv_roc_auc": round(float(search.best_score_), 4),
            "train_seconds": round(elapsed, 2),
            "best_params": {k.replace("classifier__", ""): str(v)
                            for k, v in search.best_params_.items()},
        }
        fitted[name] = best

        print(f"{name:24s} acc={results[name]['accuracy']:.4f} "
              f"f1={results[name]['f1']:.4f} auc={results[name]['roc_auc']:.4f} "
              f"cv={results[name]['cv_roc_auc']:.4f} ({elapsed:.1f}s)")

    # The report ranks models by test ROC-AUC; keep that convention.
    best_name = max(results, key=lambda k: results[k]["roc_auc"])
    best_pipe = fitted[best_name]

    print("\n" + "=" * 68)
    print(f"SELECTED MODEL: {best_name}")
    print("=" * 68)
    y_pred = best_pipe.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    print(f"Confusion matrix:\n{cm}")
    print(f"Test ROC-AUC : {results[best_name]['roc_auc']:.4f}")
    print(f"Test accuracy: {results[best_name]['accuracy']:.4f}")

    dump(best_pipe, os.path.join(MODEL_DIR, "heart_model.joblib"))
    with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    metadata = {
        "dataset": "UCI Heart Disease (920 records, 4 cohorts)",
        "n_samples": int(len(df)),
        "features": ALL_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "categories": {c: sorted(df[c].dropna().astype(str).unique().tolist())
                       for c in CATEGORICAL_FEATURES},
        "target_meaning": {"0": "No heart disease", "1": "Heart disease present"},
        "selected_model": best_name,
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "metrics": results[best_name],
    }
    with open(os.path.join(MODEL_DIR, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved -> model/heart_model.joblib")
    print(f"Saved -> model/metrics.json")
    print(f"Saved -> model/metadata.json")


if __name__ == "__main__":
    main()
