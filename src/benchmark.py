"""Quantitative benchmark pipeline for SMS spam classification.

Runs the requested modes against a stratified held-out test sample drawn from
the canonical 80/20 split (test_size=0.2, seed=42). Emits machine-readable
outputs that downstream notebooks (Phase 4) and Phase 3 visualizations consume.

Outputs:
    outputs/benchmark_results.csv      one row per mode, summary metrics
    outputs/benchmark_predictions.csv  one row per (mode, sms), incl. risk features
    outputs/benchmark_summary.md       human-readable Markdown table

Usage:
    # offline smoke (classical baseline only, no API calls)
    python -m src.benchmark --modes tfidf_lr --sample-size 200

    # cheap LLM probe (~30 OpenAI calls per LLM mode)
    python -m src.benchmark --modes tfidf_lr,base_llm,basic_rag --sample-size 30

    # full v2 run
    python -m src.benchmark \\
        --modes tfidf_lr,base_llm,basic_rag,advanced_base,advanced_lora,guarded_fallback \\
        --sample-size 100

Confusion-matrix convention (positive class = spam):
    TP = true spam, predicted spam
    FP = true ham,  predicted spam
    FN = true spam, predicted ham OR unknown
    TN = true ham,  predicted ham
    unknown predictions count as INCORRECT.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from src.classical_baselines import (
    DEFAULT_DATA_PATH,
    load_sms_data,
    stratified_split,
)
from src.features import extract_risk_features


SUPPORTED_MODES: tuple[str, ...] = (
    "tfidf_lr",
    "base_llm",
    "basic_rag",
    "advanced_base",
    "advanced_lora",
    "guarded_fallback",
)

# Modes that touch the OpenAI Chat API. Note: advanced_lora is included because
# its agentic graph still calls GPT-4o-mini for the evidence filter and the
# verifier, even though the classifier itself is the local LoRA model.
LLM_MODES: frozenset[str] = frozenset(
    {"base_llm", "basic_rag", "advanced_base", "advanced_lora", "guarded_fallback"}
)

PINECONE_MODES: frozenset[str] = frozenset(
    {"basic_rag", "advanced_base", "advanced_lora", "guarded_fallback"}
)

LORA_MODES: frozenset[str] = frozenset({"advanced_lora", "guarded_fallback"})

RISK_FEATURE_KEYS: tuple[str, ...] = (
    "has_url",
    "has_phone",
    "has_currency",
    "has_urgent_word",
    "has_prize_word",
    "has_call_instr",
    "uppercase_ratio",
    "exclamation_count",
    "message_length",
)


def normalize_prediction(text_or_label: Any) -> str:
    """Normalize any model output (raw LLM text OR pre-extracted label) into spam/ham/unknown.

    Robust to:
        - "spam" / "ham" / "SPAM" / "HAM"
        - longer LLM outputs containing "Label: spam" or "Prediction: SPAM"
        - any output containing a free-standing spam/ham token
        - missing / failed outputs (None, "")
    """
    if text_or_label is None:
        return "unknown"
    s = str(text_or_label).strip().lower()
    if not s:
        return "unknown"
    if s in {"spam", "ham"}:
        return s
    # Common LLM patterns: "Label: spam", "Prediction: spam", "Final Prediction: spam".
    m = re.search(
        r"\b(?:label|prediction|final\s*prediction)\s*:\s*(spam|ham)\b", s
    )
    if m:
        return m.group(1)
    # Fallback: take the LAST spam/ham token in the output (mirrors the
    # extraction logic in src/lora_utils.py::extract_lora_label).
    matches = re.findall(r"\b(spam|ham)\b", s)
    if matches:
        return matches[-1]
    return "unknown"


def pick_test_sample(
    test_df: pd.DataFrame, sample_size: int, seed: int
) -> pd.DataFrame:
    """Return a stratified sub-sample of the held-out test split.

    If ``sample_size <= 0`` or ``sample_size >= len(test_df)``, the full test
    split is returned. Falls back to a random sample if stratification fails
    (extremely small sample_size).
    """
    if sample_size <= 0 or sample_size >= len(test_df):
        return test_df.reset_index(drop=True)
    if sample_size < 2:
        return (
            test_df.sample(n=sample_size, random_state=seed)
            .reset_index(drop=True)
        )
    try:
        _, sampled = train_test_split(
            test_df,
            test_size=sample_size,
            stratify=test_df["label"],
            random_state=seed,
        )
    except ValueError:
        sampled = test_df.sample(n=sample_size, random_state=seed)
    return sampled.reset_index(drop=True)


def filter_modes(
    requested: list[str],
    skip_llm: bool,
    skip_lora: bool,
    skip_pinecone: bool,
) -> tuple[list[str], dict[str, str]]:
    """Filter requested modes by skip flags and unknown-mode validation."""
    skipped: dict[str, str] = {}
    final: list[str] = []
    for mode in requested:
        if mode not in SUPPORTED_MODES:
            skipped[mode] = "unknown-mode"
            continue
        if skip_llm and mode in LLM_MODES:
            skipped[mode] = "skip-llm"
            continue
        if skip_pinecone and mode in PINECONE_MODES:
            skipped[mode] = "skip-pinecone"
            continue
        if skip_lora and mode in LORA_MODES:
            skipped[mode] = "skip-lora"
            continue
        final.append(mode)
    return final, skipped


def _truncate_for_csv(text: str, limit: int = 1000) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def predict_one(
    mode: str,
    sms: str,
    baseline_pipeline: Any | None,
) -> dict[str, Any]:
    """Run a single (mode, sms) prediction.

    Never raises — failures return ``predicted_label="unknown"`` and an
    ``error_message``. Heavy dependencies (peft / langchain / pinecone) are
    imported lazily so the offline ``tfidf_lr`` mode can run on a machine
    that does not have them installed.
    """
    start = time.perf_counter()
    extras: dict[str, Any] = {
        "evidence_count": None,
        "fallback_used": None,
        "initial_lora_prediction": None,
        "lora_verification": None,
    }
    raw: str = ""

    try:
        if mode == "tfidf_lr":
            if baseline_pipeline is None:
                raise RuntimeError(
                    "tfidf_lr requested but baseline pipeline was not fitted."
                )
            raw_pred = str(baseline_pipeline.predict([sms])[0])
            normalized = normalize_prediction(raw_pred)
            raw = raw_pred

        elif mode == "base_llm":
            from src.base_modes import run_base_llm
            result = run_base_llm(sms)
            raw = result.get("raw_output") or ""
            normalized = normalize_prediction(raw)

        elif mode == "basic_rag":
            from src.base_modes import run_basic_rag
            result = run_basic_rag(sms)
            raw = result.get("raw_output") or ""
            normalized = normalize_prediction(raw)
            extras["evidence_count"] = len(result.get("retrieved_docs") or [])

        elif mode == "advanced_base":
            from src.evaluation import run_advanced_rag
            result = run_advanced_rag(sms, classifier_mode="base")
            raw = result.get("final_answer") or ""
            normalized = normalize_prediction(
                result.get("classification") or raw
            )
            extras["evidence_count"] = len(result.get("filtered_docs") or [])

        elif mode == "advanced_lora":
            from src.evaluation import run_advanced_rag
            result = run_advanced_rag(sms, classifier_mode="lora")
            raw = result.get("final_answer") or ""
            normalized = normalize_prediction(
                result.get("classification") or raw
            )
            extras["evidence_count"] = len(result.get("filtered_docs") or [])

        elif mode == "guarded_fallback":
            from src.evaluation import run_guarded_fallback
            result = run_guarded_fallback(sms)
            raw = result.get("final_answer") or ""
            normalized = normalize_prediction(
                result.get("final_prediction") or raw
            )
            extras["initial_lora_prediction"] = (
                result.get("initial_lora_prediction") or ""
            )
            extras["lora_verification"] = result.get("lora_verification") or ""
            extras["fallback_used"] = bool(result.get("fallback_used"))
            inner = (
                result.get("base_result")
                if result.get("fallback_used")
                else result.get("lora_result")
            )
            if inner:
                extras["evidence_count"] = len(
                    inner.get("filtered_docs") or []
                )

        else:
            raise ValueError(f"Unsupported mode: {mode}")

    except Exception as exc:  # noqa: BLE001
        elapsed = time.perf_counter() - start
        return {
            "predicted_label": "unknown",
            "latency_seconds": elapsed,
            "raw_output": "",
            "error_message": f"{type(exc).__name__}: {exc}",
            **extras,
        }

    elapsed = time.perf_counter() - start
    return {
        "predicted_label": normalized,
        "latency_seconds": elapsed,
        "raw_output": _truncate_for_csv(str(raw)),
        "error_message": "",
        **extras,
    }


def compute_metrics(
    y_true: list[str], y_pred: list[str], errors: int
) -> dict[str, Any]:
    """Compute spam-positive metrics. Unknown predictions count as incorrect.

    See the module docstring for the confusion-matrix convention.
    """
    n = len(y_true)
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == "spam" and p == "spam")
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == "ham" and p == "spam")
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == "spam" and p != "spam")
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == "ham" and p == "ham")

    n_true_spam = sum(1 for t in y_true if t == "spam")
    n_true_ham = sum(1 for t in y_true if t == "ham")
    n_pred_spam = sum(1 for p in y_pred if p == "spam")
    n_pred_ham = sum(1 for p in y_pred if p == "ham")

    spam_recall = tp / n_true_spam if n_true_spam > 0 else 0.0
    ham_recall = tn / n_true_ham if n_true_ham > 0 else 0.0
    spam_precision = tp / n_pred_spam if n_pred_spam > 0 else 0.0
    ham_precision = tn / n_pred_ham if n_pred_ham > 0 else 0.0

    def _f1(p: float, r: float) -> float:
        return (2 * p * r / (p + r)) if (p + r) > 0 else 0.0

    spam_f1 = _f1(spam_precision, spam_recall)
    ham_f1 = _f1(ham_precision, ham_recall)

    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    accuracy = correct / n if n else 0.0

    return {
        "n_samples": n,
        "accuracy": accuracy,
        "precision_macro": (spam_precision + ham_precision) / 2.0,
        "recall_macro": (spam_recall + ham_recall) / 2.0,
        "f1_macro": (spam_f1 + ham_f1) / 2.0,
        "spam_recall": spam_recall,
        "ham_recall": ham_recall,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "error_count": errors,
    }


def maybe_fit_baseline(
    needed: bool, train_df: pd.DataFrame
) -> Any | None:
    """Fit the TF-IDF + LR baseline only when at least one mode requires it."""
    if not needed:
        return None
    from src.classical_baselines import (
        build_tfidf_lr_pipeline,
        fit_baseline,
    )
    print("Fitting TF-IDF + Logistic Regression on train split...")
    pipe = build_tfidf_lr_pipeline()
    fit_baseline(pipe, train_df)
    return pipe


def run_benchmark(
    modes: list[str],
    sample_df: pd.DataFrame,
    baseline_pipeline: Any | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the per-mode prediction loop. Returns ``(predictions_df, results_df)``."""
    pred_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    # Cache risk features per sms_id so we don't recompute across modes.
    risk_feature_cache: dict[int, dict[str, Any]] = {}

    for mode in modes:
        print(f"\n[mode] {mode}: predicting on {len(sample_df)} sample(s)...")
        y_true: list[str] = []
        y_pred: list[str] = []
        latencies: list[float] = []
        errors = 0

        for _, row in sample_df.iterrows():
            sms_id = int(row["sms_id"])
            original_index = int(row.get("original_index", sms_id))
            text = str(row["text"])
            true_label = str(row["label"]).strip().lower()

            if sms_id not in risk_feature_cache:
                risk_feature_cache[sms_id] = extract_risk_features(text)
            features = risk_feature_cache[sms_id]

            outcome = predict_one(mode, text, baseline_pipeline)
            if outcome["error_message"]:
                errors += 1

            y_true.append(true_label)
            y_pred.append(outcome["predicted_label"])
            latencies.append(outcome["latency_seconds"])

            pred_rows.append(
                {
                    "mode": mode,
                    "sms_id": sms_id,
                    "original_index": original_index,
                    "text": text,
                    "true_label": true_label,
                    "predicted_label": outcome["predicted_label"],
                    "latency_seconds": round(outcome["latency_seconds"], 4),
                    "raw_output": outcome["raw_output"],
                    "error_message": outcome["error_message"],
                    "evidence_count": outcome["evidence_count"],
                    "fallback_used": outcome["fallback_used"],
                    "initial_lora_prediction": outcome[
                        "initial_lora_prediction"
                    ],
                    "lora_verification": outcome["lora_verification"],
                    **{k: features[k] for k in RISK_FEATURE_KEYS},
                }
            )

        metrics = compute_metrics(y_true, y_pred, errors=errors)
        avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
        summary_rows.append(
            {
                "mode": mode,
                **metrics,
                "avg_latency_s": round(avg_latency, 4),
            }
        )
        print(
            f"  {mode}: acc={metrics['accuracy']:.4f}  "
            f"f1_macro={metrics['f1_macro']:.4f}  "
            f"spam_recall={metrics['spam_recall']:.4f}  "
            f"errors={errors}"
        )

    predictions_df = pd.DataFrame(pred_rows)
    results_df = pd.DataFrame(summary_rows)
    return predictions_df, results_df


def maybe_save_confusion_matrices_png(
    output_dir: Path,
    results_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
) -> Path | None:
    """Render one confusion matrix per mode as a single PNG.

    Uses a 2x3 layout per subplot (rows = true label {ham, spam};
    cols = predicted label {ham, spam, unknown}) so failures that emit
    ``unknown`` are visible. Silently skips if matplotlib is unavailable
    or plotting raises — the rest of the benchmark always wins.
    """
    if results_df.empty or predictions_df.empty:
        return None

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: WPS433
    except ImportError as exc:
        print(f"  Skipping confusion-matrix PNG: matplotlib not available ({exc})")
        return None

    modes = results_df["mode"].tolist()
    n_modes = len(modes)
    n_cols = min(n_modes, 3)
    n_rows = (n_modes + n_cols - 1) // n_cols

    pred_label_order = ("ham", "spam", "unknown")
    true_label_order = ("ham", "spam")

    try:
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(4.2 * n_cols, 3.6 * n_rows),
            squeeze=False,
        )
        for i, mode in enumerate(modes):
            ax = axes[i // n_cols][i % n_cols]
            sub = predictions_df[predictions_df["mode"] == mode]

            cm = [[0, 0, 0], [0, 0, 0]]
            for _, prow in sub.iterrows():
                t = str(prow["true_label"]).strip().lower()
                p = str(prow["predicted_label"]).strip().lower()
                r = 0 if t == "ham" else 1
                if p == "ham":
                    c = 0
                elif p == "spam":
                    c = 1
                else:
                    c = 2
                cm[r][c] += 1

            ax.imshow(cm, cmap="Blues", aspect="auto")
            ax.set_xticks(range(len(pred_label_order)))
            ax.set_xticklabels(pred_label_order)
            ax.set_yticks(range(len(true_label_order)))
            ax.set_yticklabels(true_label_order)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")

            results_row = results_df[results_df["mode"] == mode].iloc[0]
            ax.set_title(
                f"{mode}\nn={int(results_row['n_samples'])}, "
                f"acc={results_row['accuracy']:.3f}"
            )

            max_val = max(max(row) for row in cm) or 1
            for r in range(2):
                for c in range(3):
                    val = cm[r][c]
                    color = "white" if val > max_val / 2 else "black"
                    ax.text(c, r, val, ha="center", va="center", color=color)

        # Hide any unused subplot cells.
        for j in range(n_modes, n_rows * n_cols):
            axes[j // n_cols][j % n_cols].axis("off")

        fig.tight_layout()
        out = output_dir / "confusion_matrices.png"
        fig.savefig(out, dpi=120)
        plt.close(fig)
        return out
    except Exception as exc:  # noqa: BLE001
        print(f"  Skipping confusion-matrix PNG: {type(exc).__name__}: {exc}")
        return None


def write_outputs(
    output_dir: Path,
    predictions_df: pd.DataFrame,
    results_df: pd.DataFrame,
    sample_df: pd.DataFrame,
    skipped: dict[str, str],
    seed: int,
    sample_size_arg: int,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    results_csv = output_dir / "benchmark_results.csv"
    predictions_csv = output_dir / "benchmark_predictions.csv"
    summary_md = output_dir / "benchmark_summary.md"

    results_df.to_csv(results_csv, index=False)
    predictions_df.to_csv(predictions_csv, index=False)

    spam_ratio = (
        float((sample_df["label"] == "spam").mean()) if len(sample_df) else 0.0
    )
    md_lines: list[str] = [
        "# Benchmark Summary",
        "",
        "_This benchmark uses a stratified held-out test sample. LLM/RAG modes "
        "may be run on smaller samples to control API cost._",
        "",
        "## Run config",
        "",
        "- canonical split: `test_size=0.2`, `seed=42`",
        f"- sample-size argument: {sample_size_arg} (effective: {len(sample_df)})",
        f"- random seed: {seed}",
        f"- spam ratio in sample: {spam_ratio:.3f}",
        "",
        "## Results",
        "",
    ]

    if results_df.empty:
        md_lines.append("_No modes produced results._")
    else:
        md_lines.append(
            "| mode | n | acc | P_macro | R_macro | F1_macro | "
            "spam_recall | ham_recall | TP | FP | FN | TN | "
            "avg_latency_s | errors |"
        )
        md_lines.append(
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"
        )
        for _, row in results_df.iterrows():
            md_lines.append(
                "| {mode} | {n} | {acc:.4f} | {p:.4f} | {r:.4f} | "
                "{f1:.4f} | {sr:.4f} | {hr:.4f} | {tp} | {fp} | {fn} | "
                "{tn} | {lat:.4f} | {err} |".format(
                    mode=row["mode"],
                    n=int(row["n_samples"]),
                    acc=row["accuracy"],
                    p=row["precision_macro"],
                    r=row["recall_macro"],
                    f1=row["f1_macro"],
                    sr=row["spam_recall"],
                    hr=row["ham_recall"],
                    tp=int(row["tp"]),
                    fp=int(row["fp"]),
                    fn=int(row["fn"]),
                    tn=int(row["tn"]),
                    lat=row["avg_latency_s"],
                    err=int(row["error_count"]),
                )
            )

    if skipped:
        md_lines.append("")
        md_lines.append("## Skipped modes")
        md_lines.append("")
        for mode, reason in skipped.items():
            md_lines.append(f"- `{mode}`: {reason}")

    md_lines.append("")
    summary_md.write_text("\n".join(md_lines), encoding="utf-8")

    paths: dict[str, Path] = {
        "results_csv": results_csv,
        "predictions_csv": predictions_csv,
        "summary_md": summary_md,
    }

    cm_png = maybe_save_confusion_matrices_png(
        output_dir, results_df, predictions_df
    )
    if cm_png is not None:
        paths["confusion_matrices_png"] = cm_png

    return paths


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quantitative benchmark across SMS spam classification modes."
    )
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--sample-size", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--modes",
        type=str,
        default="tfidf_lr",
        help=(
            "Comma-separated mode names. Supported: "
            + ", ".join(SUPPORTED_MODES)
            + "."
        ),
    )
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Skip modes that call the OpenAI Chat API.",
    )
    parser.add_argument(
        "--skip-lora",
        action="store_true",
        help="Skip modes that load the SmolLM2 + LoRA adapter.",
    )
    parser.add_argument(
        "--skip-pinecone",
        action="store_true",
        help="Skip modes that query the Pinecone vector store.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    requested = [m.strip() for m in args.modes.split(",") if m.strip()]
    modes, skipped = filter_modes(
        requested,
        skip_llm=args.skip_llm,
        skip_lora=args.skip_lora,
        skip_pinecone=args.skip_pinecone,
    )
    if skipped:
        print("Skipped modes:")
        for mode, reason in skipped.items():
            print(f"  - {mode}: {reason}")

    if not modes:
        print("No modes to run. Exiting.", file=sys.stderr)
        return 1

    print(f"Loading dataset from: {args.data_path}")
    df = load_sms_data(args.data_path)
    # Capture the original full-dataset row index BEFORE the split so that
    # every CSV row can be traced back to its place in data/SMSSpamCollection.
    df = df.copy()
    df["original_index"] = df.index.astype(int)

    train_df, test_df = stratified_split(df, test_size=0.2, seed=42)
    print(f"  train rows: {len(train_df)}; test rows: {len(test_df)}")

    # Stable, traceable sms_id keyed off the test split index (0..len(test_df)-1).
    # original_index, propagated as a column above, refers to the row position
    # in the original full dataset (0..len(df)-1).
    test_df = test_df.copy()
    test_df["sms_id"] = test_df.index.astype(int)

    sample_df = pick_test_sample(test_df, args.sample_size, args.seed)
    print(
        f"Selected sample: rows={len(sample_df)}; "
        f"spam={(sample_df['label'] == 'spam').sum()}; "
        f"ham={(sample_df['label'] == 'ham').sum()}"
    )

    needs_baseline = "tfidf_lr" in modes
    baseline_pipeline = maybe_fit_baseline(needs_baseline, train_df)

    predictions_df, results_df = run_benchmark(
        modes=modes,
        sample_df=sample_df,
        baseline_pipeline=baseline_pipeline,
    )

    paths = write_outputs(
        output_dir=args.output_dir,
        predictions_df=predictions_df,
        results_df=results_df,
        sample_df=sample_df,
        skipped=skipped,
        seed=args.seed,
        sample_size_arg=args.sample_size,
    )

    print("\nWrote outputs:")
    for label, path in paths.items():
        print(f"  {label}: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
