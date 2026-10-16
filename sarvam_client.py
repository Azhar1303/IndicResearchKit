"""Small, synchronous client for Sarvam's text translation REST API."""

import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

import requests


TRANSLATE_URL = "https://api.sarvam.ai/translate"
MAX_TEXT_LENGTH = 2000
MAX_ATTEMPTS = 3
MAX_RETRY_DELAY = 5.0
LANGUAGE_CODES = {"hi-IN", "od-IN", "te-IN", "ta-IN", "bn-IN", "mr-IN"}


class TranslationError(Exception):
    """A safe user-facing failure; fatal errors should stop the current run."""

    def __init__(self, message, fatal=False):
        super().__init__(message)
        self.fatal = fatal


def _retry_delay(retry_after, attempt):
    """Honor Retry-After seconds or an HTTP date, with a short bounded wait."""
    delay = 2 ** attempt
    if retry_after:
        try:
            delay = float(retry_after)
        except (TypeError, ValueError):
            try:
                retry_date = parsedate_to_datetime(retry_after)
                if retry_date.tzinfo is None:
                    retry_date = retry_date.replace(tzinfo=timezone.utc)
                delay = (retry_date - datetime.now(timezone.utc)).total_seconds()
            except (TypeError, ValueError, OverflowError):
                pass
    return min(MAX_RETRY_DELAY, max(0.0, delay))


class SarvamTranslator:
    """Reuse one HTTP session for a translation run; no credentials are logged."""

    def __init__(self, api_key):
        if not isinstance(api_key, str) or not api_key.strip():
            raise TranslationError(
                "Set SARVAM_API_KEY in your .env file before translating.",
                fatal=True,
            )
        self._api_key = api_key.strip()
        self._session = requests.Session()

    def translate(self, text, source_language_code, target_language_code):
        if not isinstance(text, str) or not text.strip():
            raise TranslationError("Translation requires non-empty text.")
        if len(text) > MAX_TEXT_LENGTH:
            raise TranslationError(
                "Text exceeds Sarvam's 2,000-character translation limit."
            )
        if (
            source_language_code not in LANGUAGE_CODES
            or target_language_code not in LANGUAGE_CODES
        ):
            raise TranslationError("An unsupported translation language was selected.")

        payload = {
            "input": text,
            "source_language_code": source_language_code,
            "target_language_code": target_language_code,
            "model": "sarvam-translate:v1",
            "mode": "formal",
        }
        for attempt in range(MAX_ATTEMPTS):
            try:
                response = self._session.post(
                    TRANSLATE_URL,
                    headers={"api-subscription-key": self._api_key},
                    json=payload,
                    timeout=(5, 30),
                    allow_redirects=False,
                )
            except requests.Timeout:
                raise TranslationError(
                    "The translation request timed out. Please try again."
                ) from None
            except requests.RequestException:
                raise TranslationError(
                    "Could not reach Sarvam. Check your connection and try again."
                ) from None

            status = response.status_code
            if status in (401, 402, 403):
                raise TranslationError(
                    "Sarvam rejected access. Check your API key, permissions, "
                    "and account credits.",
                    fatal=True,
                )
            if status == 429 or 500 <= status <= 599:
                if attempt < MAX_ATTEMPTS - 1:
                    time.sleep(_retry_delay(response.headers.get("Retry-After"), attempt))
                    continue
                message = (
                    "Sarvam's rate limit was reached. Please try again later."
                    if status == 429
                    else "Sarvam is temporarily unavailable. Please try again later."
                )
                raise TranslationError(message)
            if not 200 <= status <= 299:
                raise TranslationError(
                    "Sarvam could not translate this sample (HTTP {}).".format(status)
                )

            try:
                result = response.json()
            except ValueError:
                raise TranslationError(
                    "Sarvam returned an invalid translation response. Please try again."
                ) from None
            translated_text = result.get("translated_text") if isinstance(result, dict) else None
            if not isinstance(translated_text, str) or not translated_text.strip():
                raise TranslationError(
                    "Sarvam returned an empty or invalid translation. Please try again."
                )
            return translated_text

    def close(self):
        self._session.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
