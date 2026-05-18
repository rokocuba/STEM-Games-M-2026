import argparse
from pathlib import Path

import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="tf-logreg/reddit_ai_detector.joblib",
        help="Path to trained model",
    )
    parser.add_argument(
        "--data",
        required=True,
        help="Path to external CSV with comment,label columns",
    )
    args = parser.parse_args()

    model_path = Path(args.model)
    data_path = Path(args.data)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    model = joblib.load(model_path)

    df = pd.read_csv(data_path)

    required_columns = {"comment", "label"}
    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {missing}. "
            f"Your CSV must contain columns named: comment,label"
        )

    df = df.dropna(subset=["comment", "label"])

    X = df["comment"].astype(str)
    y_true = df["label"].astype(str)

    print("External test dataset size:", len(df))
    print("\nLabel distribution:")
    print(y_true.value_counts())

    y_pred = model.predict(X)

    print("\nAccuracy:")
    print(accuracy_score(y_true, y_pred))

    print("\nClassification report:")
    print(classification_report(y_true, y_pred))

    print("\nConfusion matrix:")
    print(confusion_matrix(y_true, y_pred))


if __name__ == "__main__":
    main()
