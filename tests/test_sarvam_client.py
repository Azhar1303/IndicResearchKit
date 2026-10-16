import unittest
from unittest.mock import Mock, patch

import requests

from sarvam_client import SarvamTranslator, TranslationError, TRANSLATE_URL


def response(status=200, translated_text="வணக்கம்", headers=None):
    result = Mock(status_code=status, headers=headers or {})
    result.json.return_value = {"translated_text": translated_text}
    return result


class SarvamTranslatorTests(unittest.TestCase):
    def setUp(self):
        session_patch = patch("sarvam_client.requests.Session")
        self.addCleanup(session_patch.stop)
        self.session = session_patch.start().return_value
        self.client = SarvamTranslator("secret-test-key")

    def test_official_payload_and_context_manager(self):
        self.session.post.return_value = response()
        with self.client as client:
            self.assertEqual(client.translate("नमस्ते", "hi-IN", "ta-IN"), "வணக்கம்")
        self.session.post.assert_called_once_with(
            TRANSLATE_URL,
            headers={"api-subscription-key": "secret-test-key"},
            json={
                "input": "नमस्ते",
                "source_language_code": "hi-IN",
                "target_language_code": "ta-IN",
                "model": "sarvam-translate:v1",
                "mode": "formal",
            },
            timeout=(5, 30),
            allow_redirects=False,
        )
        self.session.close.assert_called_once()

    @patch("sarvam_client.time.sleep")
    def test_retries_rate_limit_and_server_error_with_bounded_wait(self, sleep):
        self.session.post.side_effect = [
            response(429, headers={"Retry-After": "999"}),
            response(503),
            response(),
        ]
        self.assertEqual(self.client.translate("नमस्ते", "hi-IN", "ta-IN"), "வணக்கம்")
        self.assertEqual(self.session.post.call_count, 3)
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [5.0, 2])

    @patch("sarvam_client.time.sleep")
    def test_retries_stop_after_three_attempts(self, sleep):
        self.session.post.return_value = response(503)
        with self.assertRaises(TranslationError) as caught:
            self.client.translate("नमस्ते", "hi-IN", "ta-IN")
        self.assertFalse(caught.exception.fatal)
        self.assertEqual(self.session.post.call_count, 3)
        self.assertEqual(sleep.call_count, 2)

    def test_access_errors_are_fatal_and_never_expose_response_or_key(self):
        for status in (401, 402, 403):
            with self.subTest(status=status):
                result = response(status)
                result.text = "secret-test-key private server response"
                self.session.post.reset_mock()
                self.session.post.return_value = result
                with self.assertRaises(TranslationError) as caught:
                    self.client.translate("नमस्ते", "hi-IN", "ta-IN")
                self.assertTrue(caught.exception.fatal)
                self.assertNotIn("secret-test-key", str(caught.exception))
                self.assertNotIn("private server response", str(caught.exception))
                self.session.post.assert_called_once()
                result.json.assert_not_called()

    def test_network_and_timeout_errors_are_safe(self):
        for error in (requests.Timeout, requests.ConnectionError):
            with self.subTest(error=error):
                self.session.post.side_effect = error("secret-test-key private details")
                with self.assertRaises(TranslationError) as caught:
                    self.client.translate("नमस्ते", "hi-IN", "ta-IN")
                self.assertNotIn("secret-test-key", str(caught.exception))
                self.assertFalse(caught.exception.fatal)

    def test_rejects_invalid_responses(self):
        for value in (None, "", "  ", 123):
            with self.subTest(value=value):
                self.session.post.return_value = response(translated_text=value)
                with self.assertRaises(TranslationError):
                    self.client.translate("नमस्ते", "hi-IN", "ta-IN")
        self.session.post.return_value.json.side_effect = ValueError("secret-test-key")
        with self.assertRaises(TranslationError) as caught:
            self.client.translate("नमस्ते", "hi-IN", "ta-IN")
        self.assertNotIn("secret-test-key", str(caught.exception))

    def test_validates_inputs_before_request(self):
        for text in ("", "  ", None, "अ" * 2001):
            with self.subTest(text_length=len(text) if text else 0):
                with self.assertRaises(TranslationError):
                    self.client.translate(text, "hi-IN", "ta-IN")
        with self.assertRaises(TranslationError):
            self.client.translate("नमस्ते", "hi-IN", "invalid")
        self.session.post.assert_not_called()

    def test_missing_api_key_is_fatal(self):
        with self.assertRaises(TranslationError) as caught:
            SarvamTranslator(" ")
        self.assertTrue(caught.exception.fatal)


if __name__ == "__main__":
    unittest.main()
