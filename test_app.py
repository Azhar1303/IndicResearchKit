"""Exercise the Streamlit workflow without credentials or network access."""

import os
from pathlib import Path
import unittest
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from research import train_classifier
from sarvam_client import TranslationError


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


class AppTests(unittest.TestCase):
    def setUp(self):
        self.start_patch(patch.dict(os.environ, {"SARVAM_API_KEY": "fake-key-for-tests"}))
        self.start_patch(patch("dotenv.dotenv_values", return_value={}))
        self.http_session = self.start_patch(patch("sarvam_client.requests.Session"))
        self.train = self.start_patch(patch("research.train_classifier", wraps=train_classifier))
        self.translate = self.start_patch(
            patch(
                "sarvam_client.SarvamTranslator.translate",
                autospec=True,
                side_effect=lambda client, text, source, target: text,
            )
        )

    def start_patch(self, patcher):
        mocked = patcher.start()
        self.addCleanup(patcher.stop)
        return mocked

    def app(self):
        app = AppTest.from_file(str(APP_PATH), default_timeout=30).run()
        self.assertEqual(len(app.exception), 0)
        return app

    def run_experiment(self, app):
        app.button[0].click().run()
        self.assertEqual(len(app.exception), 0)
        return app.session_state["results"]

    @staticmethod
    def metrics(app):
        return {metric.label: metric.value for metric in app.metric}

    def assert_no_network(self):
        self.http_session.return_value.post.assert_not_called()

    def test_sample_loads_without_key_and_run_is_disabled(self):
        with patch.dict(os.environ, {"SARVAM_API_KEY": ""}):
            app = self.app()
            self.assertEqual(app.title[0].value, "IndicResearchKit")
            self.assertTrue(app.button[0].disabled)
            self.assertGreater(int(self.metrics(app)["Hindi examples"]), 1)
            self.assertEqual(self.metrics(app)["Label classes"], "2")
            self.assertEqual(app.multiselect[0].value, ["Odia"])
            self.assertTrue(any("SARVAM_API_KEY" in warning.value for warning in app.warning))
            self.train.assert_not_called()
            self.translate.assert_not_called()
            self.assert_no_network()

    def test_sample_round_trip_produces_results_and_downloads(self):
        app = self.app()
        results = self.run_experiment(app)
        self.assertEqual(len(results), 6)
        self.assertTrue(results["status"].eq("completed").all())
        self.assertTrue(results["consistent"].all())
        self.assertTrue(results["original_text"].equals(results["back_translated_text"]))
        self.assertEqual(self.metrics(app)["Prediction consistency"], "100.0%")
        self.assertEqual(self.metrics(app)["Round-trip coverage"], "100.0%")
        self.assertTrue(any("not accuracy" in message.value for message in app.info))
        download_labels = [element.proto.label for element in app.get("download_button")]
        self.assertIn("Download evaluation CSV", download_labels)
        self.assertIn("Download language summary CSV", download_labels)
        self.assertIn("Odia · 6 translated rows", download_labels)
        self.assertEqual(self.translate.call_count, 12)
        self.train.assert_called_once()
        self.assert_no_network()

    def test_review_controls_do_not_repeat_work_and_new_languages_reuse_model(self):
        app = self.app()
        results = self.run_experiment(app).copy(deep=True)
        model = app.session_state["trained_model"]

        app.slider[0].set_value(0.95).run()
        app.checkbox[0].uncheck().run()
        app.selectbox[0].select("Odia").run()
        self.assertEqual(len(app.exception), 0)
        self.assertTrue(app.session_state["results"].equals(results))
        self.assertIs(app.session_state["trained_model"], model)
        self.train.assert_called_once()
        self.assertEqual(self.translate.call_count, 12)

        app.multiselect[0].set_value(["Tamil", "Bengali"]).run()
        self.assertEqual(self.translate.call_count, 12)
        self.assertTrue(any("previous run" in message.value for message in app.info))
        updated = self.run_experiment(app)
        self.assertEqual(set(updated["language"]), {"Tamil", "Bengali"})
        self.assertEqual(len(updated), 12)
        self.assertEqual(self.translate.call_count, 36)
        self.assertIs(app.session_state["trained_model"], model)
        self.train.assert_called_once()
        self.assert_no_network()

    def test_fatal_api_error_reports_zero_coverage_and_unavailable_consistency(self):
        self.translate.side_effect = TranslationError("Check your API key and credits.", fatal=True)
        app = self.app()
        results = self.run_experiment(app)
        self.assertEqual(len(results), 6)
        self.assertEqual(results["status"].tolist(), ["forward_error"] + ["skipped"] * 5)
        self.assertTrue(results["consistent"].isna().all())
        self.assertTrue(results["back_prediction"].eq("").all())
        self.assertEqual(self.metrics(app)["Round-trip coverage"], "0.0%")
        self.assertEqual(self.metrics(app)["Prediction consistency"], "—")
        self.assertTrue(any("consistency is unavailable" in error.value for error in app.error))
        translated_download = next(
            element for element in app.get("download_button")
            if element.proto.label.startswith("Odia ·")
        )
        self.assertTrue(translated_download.proto.disabled)
        self.translate.assert_called_once()
        self.assert_no_network()


if __name__ == "__main__":
    unittest.main()
