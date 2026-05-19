import argparse
import math
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import logging as transformers_logging


DEFAULT_MODEL = "gpt2"
DEFAULT_THRESHOLD = 65.0

transformers_logging.set_verbosity_error()


def load_model(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    model.eval()
    return tokenizer, model


def calculate_perplexity(text, tokenizer, model):
    text = text.strip()

    if not text:
        raise ValueError("Text cannot be empty")

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)

    if inputs["input_ids"].shape[1] < 2:
        raise ValueError("Text is too short to score reliably")

    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])

    return math.exp(outputs.loss.item())


def predict_label(perplexity, threshold):
    # GPT-2 is usually less surprised by polished, generic AI-style text.
    label = "AI" if perplexity < threshold else "HUMAN"
    distance = abs(perplexity - threshold)
    confidence = min(0.99, 0.5 + distance / threshold)
    return label, confidence


def read_text(args):
    if args.text:
        return args.text

    if not sys.stdin.isatty():
        return sys.stdin.read()

    raise ValueError("Pass comment text with --text or pipe it through stdin")


def main():
    parser = argparse.ArgumentParser(
        description="Predict whether a comment was written by AI or a human using GPT-2 perplexity."
    )
    parser.add_argument("--text", help="Comment text to classify")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Hugging Face causal language model to use (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=(
            "Perplexity cutoff. Scores below this are predicted as AI; "
            f"scores at or above it are predicted as HUMAN (default: {DEFAULT_THRESHOLD})."
        ),
    )
    args = parser.parse_args()

    if args.threshold <= 0:
        raise ValueError("Threshold must be greater than 0")

    text = read_text(args)
    tokenizer, model = load_model(args.model)
    perplexity = calculate_perplexity(text, tokenizer, model)
    label, confidence = predict_label(perplexity, args.threshold)

    print(f"Prediction: {label}")
    print(f"Confidence: {confidence:.2%}")
    print(f"Perplexity: {perplexity:.2f}")
    print(f"Threshold: {args.threshold:.2f}")


if __name__ == "__main__":
    try:
        main()
    except ValueError as error:
        raise SystemExit(f"Error: {error}") from error
