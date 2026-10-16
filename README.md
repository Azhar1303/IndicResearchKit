# IndicResearchKit

A small Streamlit prototype for exploring whether a fixed Hindi text classifier gives consistent predictions after translation into another Indian language and back to Hindi. Supports Odia, Telugu, Tamil, Bengali, and Marathi through Sarvam AI.

## Run locally

Use Python 3.9 or newer (3.11 recommended).

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
# Edit .env and set SARVAM_API_KEY to your Sarvam API key.
streamlit run app.py
```

On Windows, activate with `.venv\Scripts\activate`. Keep `.env` private; the API key is read from `SARVAM_API_KEY` and is never included in exports.

## Try it

1. Use the bundled `data/sample_hindi.csv`, or upload a UTF-8 CSV with `text` and `label` columns.
2. Select one or more target languages and the number of source examples to evaluate. The default is the first six rows to keep API usage small.
3. Run the experiment, review consistency, coverage, and flagged samples, then download translated datasets and evaluation CSVs.

The classifier uses character TF-IDF features and logistic regression. It trains once on the **entire original Hindi dataset**; the row limit only controls translation and evaluation. The same fitted classifier predicts each original Hindi text and each Hindi back-translation. It is never retrained on translated data.

Labels are preserved as strings, including values such as `NA` and `001`. Input must have nonblank text and labels and at least two label classes. Blank values, malformed CSVs, missing columns, text longer than 2,000 characters, and duplicate Hindi texts with conflicting labels are rejected.

## Read the results correctly

- **Prediction consistency** is the fraction of successful round trips where the original prediction equals the back-translated prediction. It is **not accuracy**, a translation-quality score, or proof that a model works in another language. A consistently wrong prediction still counts as consistent.
- **Coverage** shows how many requested examples completed their round trip. API failures are reported and excluded from the consistency denominator. Read consistency alongside coverage.
- Samples are flagged when predictions change, either prediction score is below the selected threshold (default `0.60`), or translation is incomplete. Scores are uncalibrated; the threshold is a review heuristic, not a reliability guarantee. Changing the threshold does not repeat API calls.
- No held-out accuracy is reported. The source labels are retained for research use, but translated labels have not been verified by speakers of the target languages.

The balanced positive/negative sentiment sample is a tiny, artificial demonstration dataset. It provides no evidence of generalization.

## API usage and exports

The app calls Sarvam's `POST /translate` endpoint with `sarvam-translate:v1`, once for the forward translation and once for the Hindi back-translation. Evaluating `N` rows in `L` languages uses up to `2 × N × L` base requests, plus bounded retries for transient failures. Your dataset text is sent to Sarvam's external API, so use data you are authorized to share and account for your API plan's charges and limits.

Official references: [Translate Text API](https://docs.sarvam.ai/api-reference/text/translate-text) and [Sarvam Translate model](https://docs.sarvam.ai/api/getting-started/models/sarvam-translate).

Errors are shown without discarding completed work. Per-language translated CSVs contain `text`, `label`, and `source_row` for successful forward translations, including rows whose later back-translation failed. Evaluation exports include per-example results and a per-language summary. A failed translation is never substituted with a fabricated example.

## Checks

```sh
python -m unittest discover -s tests
```

Tests use mocked API responses; a live end-to-end run requires a valid Sarvam API key.
