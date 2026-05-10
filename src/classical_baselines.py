"""TF-IDF + Logistic Regression classical baseline for SMS spam classification.

Public API:
    - load_sms_data(path)
    - stratified_split(df, test_size, seed)
    - build_tfidf_lr_pipeline()
    - fit_baseline(pipeline, train_df)
    - predict_baseline(pipeline, texts)
    - evaluate_predictions(y_true, y_pred)
    - save_baseline(pipeline, path)
    - load_baseline(path)
    - main()  # entry point: `python -m src.classical_baselines`
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


DEFAULT_DATA_PATH = Path("data/SMSSpamCollection")
DEFAULT_MODEL_PATH = Path("outputs/baseline_tfidf_lr.joblib")
LABELS: tuple[str, ...] = ("ham", "spam")


def load_sms_data(path: Path | str = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Read the UCI SMS Spam Collection file into a DataFrame.

    The file is a tab-separated two-column file: <label>\\t<text>.
    Returns a DataFrame with normalized lower-case ``label`` and string
    ``text`` columns.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"SMS dataset not found at: {path}")

    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["label", "text"],
        encoding="utf-8",
        quoting=3,
    )
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df["text"] = df["text"].astype(str)
    return df


def stratified_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified train/test split preserving the spam/ham ratio."""
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df["label"],
        random_state=seed,
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def build_tfidf_lr_pipeline() -> Pipeline:
    """TF-IDF (1-2 ngrams) + class-weighted Logistic Regression."""
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=2,
                    sublinear_tf=True,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                ),
            ),
        ]
    )


def fit_baseline(pipeline: Pipeline, train_df: pd.DataFrame) -> Pipeline:
    pipeline.fit(train_df["text"].tolist(), train_df["label"].tolist())
    return pipeline


def predict_baseline(pipeline: Pipeline, texts: list[str]) -> list[str]:
    return [str(label) for label in pipeline.predict(texts)]


def evaluate_predictions(
    y_true: list[str],
    y_pred: list[str],
) -> dict[str, Any]:
    """Compute the metric dict shared with the (later) benchmark pipeline."""
    cm = confusion_matrix(y_true, y_pred, labels=list(LABELS))
    # rows = true, cols = pred. Order = (ham, spam).
    tn, fp = int(cm[0, 0]), int(cm[0, 1])
    fn, tp = int(cm[1, 0]), int(cm[1, 1])
    spam_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    ham_recall = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(
            precision_score(
                y_true,
                y_pred,
                labels=list(LABELS),
                average="macro",
                zero_division=0,
            )
        ),
        "recall_macro": float(
            recall_score(
                y_true,
                y_pred,
                labels=list(LABELS),
                average="macro",
                zero_division=0,
            )
        ),
        "f1_macro": float(
            f1_score(
                y_true,
                y_pred,
                labels=list(LABELS),
                average="macro",
                zero_division=0,
            )
        ),
        "spam_recall": spam_recall,
        "ham_recall": ham_recall,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def save_baseline(pipeline: Pipeline, path: Path | str = DEFAULT_MODEL_PATH) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    return path


def load_baseline(path: Path | str = DEFAULT_MODEL_PATH) -> Pipeline:
    return joblib.load(path)


def _format_metrics(metrics: dict[str, Any]) -> str:
    return (
        f"  accuracy:        {metrics['accuracy']:.4f}\n"
        f"  precision_macro: {metrics['precision_macro']:.4f}\n"
        f"  recall_macro:    {metrics['recall_macro']:.4f}\n"
        f"  f1_macro:        {metrics['f1_macro']:.4f}\n"
        f"  spam_recall:     {metrics['spam_recall']:.4f}\n"
        f"  ham_recall:      {metrics['ham_recall']:.4f}\n"
        f"  TP/FP/FN/TN:     {metrics['tp']}/{metrics['fp']}/{metrics['fn']}/{metrics['tn']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and evaluate the TF-IDF + Logistic Regression SMS baseline."
    )
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Skip writing the joblib artifact (useful for smoke tests).",
    )
    args = parser.parse_args()

    print(f"Loading dataset from: {args.data_path}")
    df = load_sms_data(args.data_path)
    counts = df["label"].value_counts().to_dict()
    print(f"  rows: {len(df)}; label counts: {counts}")

    train_df, test_df = stratified_split(
        df, test_size=args.test_size, seed=args.seed
    )
    print(f"  train rows: {len(train_df)}; test rows: {len(test_df)}")

    pipeline = build_tfidf_lr_pipeline()
    print("Fitting TF-IDF + Logistic Regression pipeline...")
    fit_baseline(pipeline, train_df)

    print("Evaluating on held-out test split...")
    y_true = test_df["label"].tolist()
    y_pred = predict_baseline(pipeline, test_df["text"].tolist())
    metrics = evaluate_predictions(y_true, y_pred)
    print("Test metrics:")
    print(_format_metrics(metrics))

    if not args.no_save:
        saved = save_baseline(pipeline, args.model_path)
        print(f"Saved baseline to: {saved}")
    else:
        print("Skipped saving (--no-save).")


if __name__ == "__main__":
    main()
