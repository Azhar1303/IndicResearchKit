"""Offline checks for dataset integrity and fixed-model evaluation semantics."""

import io
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from sklearn.pipeline import Pipeline

from research import (
    csv_bytes,
    evaluate_dataset,
    load_dataset,
    review_samples,
    summarize_results,
    train_classifier,
    translated_dataset,
)
from sarvam_client import TranslationError


GOOD = "अच्छा अनुभव"
BAD = "बुरा अनुभव"
GOOD_BACK = "अच्छा अनुभव वापस"
CHANGED_BACK = "अब अच्छा अनुभव"


class FixedClassifier:
    """Predictions are known in advance; fitting during evaluation is an error."""

    classes_ = np.array(["negative", "positive"])
    probabilities = {
        GOOD: [0.1, 0.9],
        BAD: [0.8, 0.2],
        GOOD_BACK: [0.25, 0.75],
        CHANGED_BACK: [0.4, 0.6],
    }

    def fit(self, *args, **kwargs):
        raise AssertionError("Evaluation must never fit the classifier.")

    def predict_proba(self, texts):
        return np.array([self.probabilities[text] for text in texts])

    def predict(self, texts):
        return self.classes_[self.predict_proba(texts).argmax(axis=1)]


class FakeTranslator:
    def __init__(self, failures=None):
        self.failures = failures or {}
        self.calls = []

    def translate(self, text, source_language_code, target_language_code):
        call = (text, source_language_code, target_language_code)
        self.calls.append(call)
        if call in self.failures:
            raise self.failures[call]
        if source_language_code == "hi-IN":
            return target_language_code + ":" + text
        if text == source_language_code + ":" + GOOD:
            return GOOD_BACK
        if text == source_language_code + ":" + BAD:
            return CHANGED_BACK
        raise AssertionError("Unexpected translation request: {!r}".format(call))


def sample_data():
    return pd.DataFrame({"text": [GOOD, BAD], "label": ["positive", "negative"]})


class DatasetTests(unittest.TestCase):
    def test_load_preserves_strings_and_accepts_bom_extra_columns_and_quotes(self):
        content = (
            '\ufefftext,label,ignored\n" नमस्ते, दुनिया ",001,x\n'
            '"दूसरी\nपंक्ति",NA,y\nतीसरा,null,z\n'
        ).encode("utf-8")
        data = load_dataset(content)
        self.assertEqual(list(data.columns), ["text", "label"])
        self.assertEqual(data["text"].tolist(), [" नमस्ते, दुनिया ", "दूसरी\nपंक्ति", "तीसरा"])
        self.assertEqual(data["label"].tolist(), ["001", "NA", "null"])
        self.assertEqual(data.index.tolist(), [0, 1, 2])

    def test_load_rejects_ambiguous_csv_and_missing_required_headers(self):
        invalid = [
            "text,label\nअच्छा,positive,extra\nबुरा,negative\n",
            "text,label\nअच्छा\nबुरा,negative\n",
            "text,label,label\nअच्छा,p,p\nबुरा,n,n\n",
            'text,label\n"अच्छा,positive\nबुरा,negative\n',
            "message,label\nअच्छा,p\nबुरा,n\n",
            "text,category\nअच्छा,p\nबुरा,n\n",
        ]
        for content in invalid:
            with self.subTest(content=content):
                with self.assertRaises(ValueError):
                    load_dataset(content.encode("utf-8"))

    def test_load_rejects_blank_values_one_class_oversize_and_label_conflicts(self):
        invalid = [
            "text,label\n ,positive\nबुरा,negative\n",
            "text,label\nअच्छा, \nबुरा,negative\n",
            "text,label\nअच्छा,positive\nबुरा,positive\n",
            "text,label\n" + "अ" * 2001 + ",positive\nबुरा,negative\n",
            "text,label\nअच्छा,positive\n अच्छा ,negative\n",
        ]
        for content in invalid:
            with self.subTest(content=content[:80]):
                with self.assertRaises(ValueError):
                    load_dataset(content.encode("utf-8"))

    def test_sample_trains_and_predicts_with_a_real_sklearn_pipeline(self):
        sample_path = Path(__file__).resolve().parents[1] / "data" / "sample_hindi.csv"
        data = load_dataset(sample_path.read_bytes())
        classifier = train_classifier(data)
        self.assertIsInstance(classifier, Pipeline)
        predictions = classifier.predict(data["text"])
        probabilities = classifier.predict_proba(data["text"])
        self.assertEqual(len(predictions), len(data))
        self.assertEqual(set(classifier.classes_), set(data["label"]))
        self.assertTrue(set(predictions).issubset(set(data["label"])))
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)


class EvaluationTests(unittest.TestCase):
    def test_fixed_model_round_trip_preserves_labels_and_reports_progress(self):
        data = sample_data()
        # Both original predictions are wrong against these labels; consistency
        # must still compare the two predictions, rather than ground truth.
        data["label"] = ["negative", "positive"]
        original = data.copy(deep=True)
        translator = FakeTranslator()
        progress = []
        results = evaluate_dataset(
            FixedClassifier(), data, ["Tamil", "Telugu"], translator,
            progress=lambda completed, total: progress.append((completed, total)),
        )
        self.assertEqual(len(results), 4)
        self.assertEqual(set(results["status"]), {"completed"})
        self.assertEqual(progress, [(1, 4), (2, 4), (3, 4), (4, 4)])
        self.assertEqual(len(translator.calls), 8)
        self.assertTrue({
            "source_row", "language", "language_code", "label", "original_text",
            "translated_text", "back_translated_text", "original_prediction",
            "original_confidence", "back_prediction", "back_confidence",
            "consistent", "status", "error",
        }.issubset(results.columns))
        for language, code in (("Tamil", "ta-IN"), ("Telugu", "te-IN")):
            rows = results[results["language"] == language].sort_values("source_row")
            self.assertEqual(rows["language_code"].tolist(), [code, code])
            self.assertEqual(rows["source_row"].tolist(), [1, 2])
            self.assertEqual(rows["label"].tolist(), ["negative", "positive"])
            self.assertEqual(rows["original_prediction"].tolist(), ["positive", "negative"])
            self.assertEqual(rows["back_prediction"].tolist(), ["positive", "positive"])
            self.assertEqual(rows["consistent"].tolist(), [True, False])
            np.testing.assert_allclose(rows["original_confidence"], [0.9, 0.8])
            np.testing.assert_allclose(rows["back_confidence"], [0.75, 0.6])
        assert_frame_equal(data, original)

    def test_sample_failures_continue_and_forward_success_remains_downloadable(self):
        translator = FakeTranslator({
            (GOOD, "hi-IN", "ta-IN"): TranslationError("Forward unavailable"),
            ("ta-IN:" + BAD, "ta-IN", "hi-IN"): TranslationError("Back unavailable"),
        })
        results = evaluate_dataset(FixedClassifier(), sample_data(), ["Tamil", "Telugu"], translator)
        tamil = results[results["language"] == "Tamil"].sort_values("source_row")
        self.assertEqual(tamil["status"].tolist(), ["forward_error", "back_error"])
        self.assertTrue(tamil["consistent"].isna().all())
        self.assertTrue(tamil["error"].str.len().gt(0).all())
        self.assertEqual(str(results["consistent"].dtype), "boolean")
        self.assertEqual(results[results["language"] == "Telugu"]["status"].tolist(), ["completed", "completed"])
        self.assertNotIn(("ta-IN:" + GOOD, "ta-IN", "hi-IN"), translator.calls)
        downloaded = translated_dataset(results, "Tamil")
        self.assertEqual(list(downloaded.columns), ["text", "label", "source_row"])
        self.assertEqual(downloaded.to_dict("records"), [
            {"text": "ta-IN:" + BAD, "label": "negative", "source_row": 2}
        ])

    def test_fatal_error_stops_all_further_requests_but_keeps_expected_rows(self):
        for call, expected_status, request_count, message in (
            ((GOOD, "hi-IN", "ta-IN"), "forward_error", 1, "Access denied"),
            (("ta-IN:" + GOOD, "ta-IN", "hi-IN"), "back_error", 2, "Access denied"),
            ((GOOD, "hi-IN", "ta-IN"), "forward_error", 1, ""),
        ):
            with self.subTest(failing_call=call, message=message):
                translator = FakeTranslator({call: TranslationError(message, fatal=True)})
                progress = []
                results = evaluate_dataset(
                    FixedClassifier(), sample_data(), ["Tamil", "Telugu"], translator,
                    progress=lambda completed, total: progress.append((completed, total)),
                )
                self.assertEqual(len(translator.calls), request_count)
                self.assertEqual(len(results), 4)
                self.assertEqual(results["status"].tolist(), [expected_status, "skipped", "skipped", "skipped"])
                self.assertTrue(results["consistent"].isna().all())
                self.assertEqual(progress, [(1, 4), (2, 4), (3, 4), (4, 4)])


class ResultsTests(unittest.TestCase):
    def test_review_flags_both_confidences_changes_and_incomplete_rows_without_mutation(self):
        results = pd.DataFrame({
            "status": ["completed"] * 4 + ["forward_error", "back_error"],
            "consistent": pd.array([True, False, True, True, pd.NA, pd.NA], dtype="boolean"),
            "original_confidence": [0.75, 0.9, 0.7, 0.9, 0.9, 0.9],
            "back_confidence": [0.75, 0.9, 0.9, 0.7, np.nan, np.nan],
        })
        original = results.copy(deep=True)
        reviewed = review_samples(results, threshold=0.75)
        self.assertEqual(reviewed["flagged"].tolist(), [False, True, True, True, True, True])
        self.assertEqual(reviewed.iloc[0]["flag_reason"], "")
        self.assertTrue(reviewed.loc[reviewed["flagged"], "flag_reason"].str.len().gt(0).all())
        assert_frame_equal(results, original)

    def test_summary_uses_completed_pairs_as_consistency_denominator(self):
        reviewed = pd.DataFrame({
            "language": ["Tamil"] * 3 + ["Telugu"] * 2,
            "language_code": ["ta-IN"] * 3 + ["te-IN"] * 2,
            "status": ["completed", "completed", "forward_error", "back_error", "skipped"],
            "consistent": pd.array([True, False, pd.NA, pd.NA, pd.NA], dtype="boolean"),
            "flagged": [False, True, True, True, True],
        })
        summary = summarize_results(reviewed).set_index("language")
        tamil = summary.loc["Tamil"]
        self.assertEqual(tamil["language_code"], "ta-IN")
        self.assertEqual((tamil["requested"], tamil["completed"], tamil["failed"]), (3, 2, 1))
        self.assertEqual((tamil["consistent"], tamil["flagged"]), (1, 2))
        self.assertAlmostEqual(tamil["consistency_pct"], 50.0)
        self.assertAlmostEqual(tamil["coverage_pct"], 200.0 / 3)
        telugu = summary.loc["Telugu"]
        self.assertEqual((telugu["requested"], telugu["completed"], telugu["failed"]), (2, 0, 2))
        self.assertEqual(telugu["coverage_pct"], 0)
        self.assertTrue(pd.isna(telugu["consistency_pct"]))

    def test_csv_download_is_bom_utf8_and_round_trips_indic_text_and_label_strings(self):
        data = pd.DataFrame({"text": ["தமிழ், உரை", "हिन्दी\nवाक्य"], "label": ["001", "NA"]})
        encoded = csv_bytes(data)
        self.assertIsInstance(encoded, bytes)
        self.assertTrue(encoded.startswith(b"\xef\xbb\xbf"))
        decoded = pd.read_csv(io.BytesIO(encoded), dtype=str, keep_default_na=False)
        assert_frame_equal(decoded, data)


if __name__ == "__main__":
    unittest.main()
