import argparse
from pathlib import Path

import joblib


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="tf-logreg/reddit_ai_detector.joblib",
        help="Path to trained model",
    )
    parser.add_argument(
        "--text",
        required=True,
        help="Comment text to classify",
    )
    args = parser.parse_args()

    model_path = Path(args.model)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = joblib.load(model_path)

    comment = [args.text]

    prediction = model.predict(comment)[0]
    probabilities = model.predict_proba(comment)[0]
    classes = model.named_steps["classifier"].classes_

    print("Prediction:", prediction)
    print("\nProbabilities:")

    for label, probability in zip(classes, probabilities):
        print(f"{label}: {probability:.4f}")


if __name__ == "__main__":
    main()
