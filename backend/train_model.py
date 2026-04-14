import json
from pathlib import Path

import joblib
from sklearn import metrics
from sklearn.model_selection import train_test_split

from data_loader import load_dataset
from features import build_features
from modeling import build_pipeline


ROOT_DIR = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = ROOT_DIR / "backend" / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "fake_news_pipeline.joblib"
METRICS_PATH = ARTIFACTS_DIR / "training_metrics.json"


def train_and_export() -> None:
    df = load_dataset()
    features, labels = build_features(df)

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        random_state=2,
        test_size=0.3,
        stratify=labels,
    )

    pipeline = build_pipeline()
    pipeline.fit(x_train, y_train)

    y_proba = pipeline.predict_proba(x_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    scores = {
        "roc_auc": float(metrics.roc_auc_score(y_test, y_proba)),
        "classification_report": metrics.classification_report(y_test, y_pred, output_dict=True),
    }

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)

    with METRICS_PATH.open("w", encoding="utf-8") as f:
        json.dump(scores, f, indent=2)

    print(f"Model exported to: {MODEL_PATH}")
    print(f"Metrics exported to: {METRICS_PATH}")
    print(f"ROC-AUC: {scores['roc_auc']:.4f}")


if __name__ == "__main__":
    train_and_export()
