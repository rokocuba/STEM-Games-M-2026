import argparse
from pathlib import Path

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to CSV file with comment,label columns")
    parser.add_argument(
        "--model-out",
        default="tf-logreg/reddit_ai_detector.joblib",
        help="Where to save the trained model",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()

    data_path = Path(args.data)

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

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
    y = df["label"].astype(str)

    print("Dataset size:", len(df))
    print("\nLabel distribution:")
    print(y.value_counts())

    if y.nunique() < 2:
        raise ValueError("You need at least two labels, for example: human and ai")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=42,
        stratify=y,
    )

    print("\nTraining examples:", len(X_train))
    print("Testing examples:", len(X_test))

    model = Pipeline([
        ("features", FeatureUnion([
            ("word_tfidf", TfidfVectorizer(
                analyzer="word",
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95,
                sublinear_tf=True,
            )),
            ("char_tfidf", TfidfVectorizer(
                analyzer="char",
                ngram_range=(3, 5),
                min_df=2,
                max_df=0.95,
                sublinear_tf=True,
            )),
        ])),
        ("classifier", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="liblinear",
        )),
    ])

    print("\nTraining model...")
    model.fit(X_train, y_train)

    print("\nTesting model...")
    y_pred = model.predict(X_test)

    print("\nAccuracy:")
    print(accuracy_score(y_test, y_pred))

    print("\nClassification report:")
    print(classification_report(y_test, y_pred))

    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    model_out = Path(args.model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, model_out)

    print(f"\nSaved model to: {model_out}")


if __name__ == "__main__":
    main()
