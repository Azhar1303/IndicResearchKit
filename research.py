"""Small, UI-independent dataset, classifier, and round-trip evaluation helpers."""

import csv
import io
from typing import Callable, Optional

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from sarvam_client import TranslationError

LANGUAGES = {
    "Odia": "od-IN",
    "Telugu": "te-IN",
    "Tamil": "ta-IN",
    "Bengali": "bn-IN",
    "Marathi": "mr-IN",
}


def load_dataset(content: bytes) -> pd.DataFrame:
    """Validate UTF-8 CSV without coercing class names or changing source text."""
    try:
        reader = csv.reader(io.StringIO(content.decode("utf-8-sig")), strict=True)
        headers = [column.strip() for column in next(reader)]
        if len(headers) != len(set(headers)):
            raise ValueError("CSV column names must be unique.")
        if not {"text", "label"}.issubset(headers):
            raise ValueError("CSV must contain text and label columns.")
        rows = []
        for row in reader:
            if not row:  # Allow blank lines, as most CSV readers do.
                continue
            if len(row) != len(headers):
                raise ValueError(
                    f"CSV record ending at line {reader.line_num} has the wrong number of fields."
                )
            rows.append(row)
    except UnicodeDecodeError as exc:
        raise ValueError("Use a UTF-8 CSV file (UTF-8 with BOM is also supported).") from exc
    except (StopIteration, csv.Error) as exc:
        raise ValueError("The CSV is empty or malformed. Check quoting and separators.") from exc

    data = pd.DataFrame(rows, columns=headers)[["text", "label"]]
    if data.empty:
        raise ValueError("The CSV must contain Hindi examples and labels.")
    for column in ("text", "label"):
        blank = data[column].str.strip().eq("")
        if blank.any():
            raise ValueError(f"{column} is blank in data row {int(blank.idxmax()) + 1}.")
    if data["text"].str.len().gt(2000).any():
        raise ValueError("Each text must be at most 2,000 characters for Sarvam Translate.")
    if data["label"].nunique() < 2:
        raise ValueError("Include at least two distinct labels to train the classifier.")
    normalized = data.assign(normalized_text=data["text"].str.strip())
    if normalized.groupby("normalized_text")["label"].nunique().gt(1).any():
        raise ValueError("The same Hindi text has conflicting labels. Resolve these before training.")
    return data.reset_index(drop=True)


def train_classifier(data: pd.DataFrame) -> Pipeline:
    """Fit once on the original Hindi corpus; character features retain Indic marks."""
    model = Pipeline(
        [
            ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5))),
            (
                "classifier",
                LogisticRegression(C=4.0, class_weight="balanced", max_iter=1000, random_state=42),
            ),
        ]
    )
    model.fit(data["text"].tolist(), data["label"].tolist())
    return model


def evaluate_dataset(
    classifier,
    data: pd.DataFrame,
    languages: list,
    translator,
    progress: Optional[Callable[[int, int], None]] = None,
) -> pd.DataFrame:
    """Translate each sample out and back, using the same already fitted model.

    A failed round trip is recorded but has no consistency value. Fatal API errors
    stop network requests while retaining all requested rows for coverage reporting.
    """
    if data.empty or not languages:
        raise ValueError("Select at least one sample and one target language.")
    if len(languages) != len(set(languages)) or any(name not in LANGUAGES for name in languages):
        raise ValueError("Select unique target languages from the supported list.")
    original_probabilities = classifier.predict_proba(data["text"].tolist())
    original_indices = original_probabilities.argmax(axis=1)
    records = []
    stopped_reason = None
    total = len(data) * len(languages)
    for language in languages:
        for position, sample in enumerate(data.itertuples(index=False)):
            original_index = original_indices[position]
            record = {
                "source_row": position + 1,
                "language": language,
                "language_code": LANGUAGES[language],
                "label": sample.label,
                "original_text": sample.text,
                "translated_text": "",
                "back_translated_text": "",
                "original_prediction": str(classifier.classes_[original_index]),
                "original_confidence": float(original_probabilities[position, original_index]),
                "back_prediction": "",
                "back_confidence": None,
                "consistent": pd.NA,
                "status": "skipped",
                "error": "",
            }
            if stopped_reason is not None:
                record["error"] = f"Skipped after a fatal API error: {stopped_reason}"
            else:
                stage = "forward_error"
                try:
                    record["translated_text"] = translator.translate(
                        sample.text, "hi-IN", LANGUAGES[language]
                    )
                    stage = "back_error"
                    record["back_translated_text"] = translator.translate(
                        record["translated_text"], LANGUAGES[language], "hi-IN"
                    )
                    back_probabilities = classifier.predict_proba([record["back_translated_text"]])[0]
                    back_index = back_probabilities.argmax()
                    record["back_prediction"] = str(classifier.classes_[back_index])
                    record["back_confidence"] = float(back_probabilities[back_index])
                    record["consistent"] = record["original_prediction"] == record["back_prediction"]
                    record["status"] = "completed"
                except TranslationError as exc:
                    record["status"] = stage
                    record["error"] = str(exc)
                    if exc.fatal:
                        stopped_reason = str(exc)
            records.append(record)
            if progress:
                progress(len(records), total)
    results = pd.DataFrame(records)
    results["consistent"] = results["consistent"].astype("boolean")
    return results


def review_samples(results: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Apply a transparent review heuristic without repeating paid translations."""
    if not 0 <= threshold <= 1:
        raise ValueError("Confidence threshold must be between 0 and 1.")
    reviewed = results.copy()
    reasons = []
    for row in reviewed.itertuples(index=False):
        flags = []
        if row.status != "completed":
            flags.append("Translation incomplete")
        else:
            if not row.consistent:
                flags.append("Prediction changed")
            if row.original_confidence < threshold:
                flags.append("Low original confidence")
            if row.back_confidence < threshold:
                flags.append("Low back-translation confidence")
        reasons.append("; ".join(flags))
    reviewed["flag_reason"] = reasons
    reviewed["flagged"] = reviewed["flag_reason"].ne("")
    reviewed["confidence_threshold"] = threshold
    return reviewed


def summarize_results(reviewed: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (language, code), group in reviewed.groupby(["language", "language_code"], sort=False):
        completed = int(group["status"].eq("completed").sum())
        consistent = int(group["consistent"].sum())
        rows.append(
            {
                "language": language,
                "language_code": code,
                "requested": len(group),
                "completed": completed,
                "failed": len(group) - completed,
                "consistent": consistent,
                "consistency_pct": 100 * consistent / completed if completed else None,
                "coverage_pct": 100 * completed / len(group),
                "flagged": int(group["flagged"].sum()),
            }
        )
    return pd.DataFrame(rows)


def translated_dataset(results: pd.DataFrame, language: str) -> pd.DataFrame:
    """Include every successful forward translation, even if its return leg failed."""
    rows = results.loc[
        results["language"].eq(language) & results["translated_text"].ne(""),
        ["translated_text", "label", "source_row"],
    ]
    return rows.rename(columns={"translated_text": "text"}).reset_index(drop=True)


def csv_bytes(data: pd.DataFrame) -> bytes:
    """BOM makes Indian scripts display correctly in spreadsheet applications."""
    return data.to_csv(index=False).encode("utf-8-sig")
