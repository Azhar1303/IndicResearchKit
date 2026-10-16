"""Run with: streamlit run app.py"""

import hashlib
import os
from pathlib import Path

import streamlit as st
from dotenv import dotenv_values

from research import (
    LANGUAGES,
    csv_bytes,
    evaluate_dataset,
    load_dataset,
    review_samples,
    summarize_results,
    train_classifier,
    translated_dataset,
)
from sarvam_client import SarvamTranslator, TranslationError

ROOT = Path(__file__).resolve().parent
SAMPLE_PATH = ROOT / "data" / "sample_hindi.csv"


def show_results(raw_results, config, threshold):
    reviewed = review_samples(raw_results, threshold)
    summary = summarize_results(reviewed)
    completed = int(reviewed["status"].eq("completed").sum())
    consistent = int(reviewed["consistent"].sum())
    total = len(reviewed)

    st.divider()
    st.subheader("Experiment results")
    st.caption(
        f"First {config['rows']} Hindi rows → {', '.join(config['languages'])} → Hindi. "
        f"The fixed classifier was trained on all {config['training_rows']} original rows."
    )
    columns = st.columns(4)
    columns[0].metric(
        "Prediction consistency",
        f"{100 * consistent / completed:.1f}%" if completed else "—",
        help="Original and back-translated predictions agree / completed round trips. This is not accuracy.",
    )
    columns[1].metric(
        "Round-trip coverage",
        f"{100 * completed / total:.1f}%",
        help=f"{completed} of {total} requested sample-language pairs completed. Failures are excluded from consistency.",
    )
    columns[2].metric("Prediction changes", completed - consistent)
    columns[3].metric(
        "Flagged for review",
        int(reviewed["flagged"].sum()),
        help="Changed predictions, scores below the threshold, or incomplete translations.",
    )

    st.info(
        "Consistency measures prediction agreement, not accuracy or translation quality. "
        "A consistently wrong prediction still counts as consistent. "
        "This prototype reports no held-out accuracy."
    )
    incomplete = total - completed
    if incomplete:
        st.warning(
            f"{incomplete} of {total} round trips did not complete. "
            "Check status and error in Sample review. Successful forward translations remain downloadable."
        )
    if not completed:
        st.error("No round trips completed, so prediction consistency is unavailable.")

    overview, samples, downloads = st.tabs(["Overview", "Sample review", "Downloads"])
    with overview:
        st.dataframe(
            summary,
            hide_index=True,
            width="stretch",
            column_config={
                "consistency_pct": st.column_config.NumberColumn("Consistency (%)", format="%.1f"),
                "coverage_pct": st.column_config.NumberColumn("Coverage (%)", format="%.1f"),
            },
        )
        st.caption(
            "Consistency = agreeing predictions ÷ completed round trips. "
            "Coverage = completed ÷ requested. Failed counts include skipped rows."
        )
        if completed and len(summary) > 1:
            st.bar_chart(summary.set_index("language")[["consistency_pct"]], height=240)
        st.caption(
            "Source labels are copied to translated examples without verification. "
            "Agreement cannot establish whether the translation preserved the true class."
        )

    with samples:
        controls = st.columns([2, 1])
        selected = controls[0].selectbox(
            "Review language", ["All languages"] + list(config["languages"])
        )
        flagged_only = controls[1].checkbox("Only flagged samples", value=True)
        view = reviewed
        if selected != "All languages":
            view = view.loc[view["language"].eq(selected)]
        if flagged_only:
            view = view.loc[view["flagged"]]
        st.caption(
            f"Review threshold: {threshold:.2f}. Scores are uncalibrated model outputs. "
            "Flags are a review heuristic, not a reliability guarantee."
        )
        if view.empty:
            st.success("No samples match these review filters.")
        else:
            st.dataframe(
                view[
                    [
                        "source_row", "language", "label", "original_prediction", "back_prediction",
                        "consistent", "original_confidence", "back_confidence", "flag_reason",
                        "original_text", "translated_text", "back_translated_text", "status", "error",
                    ]
                ],
                hide_index=True,
                width="stretch",
                column_config={
                    "original_confidence": st.column_config.NumberColumn(format="%.3f"),
                    "back_confidence": st.column_config.NumberColumn(format="%.3f"),
                },
            )

    with downloads:
        st.caption("Exports include all experiment rows, regardless of the review filters.")
        left, right = st.columns(2)
        left.download_button(
            "Download evaluation CSV", csv_bytes(reviewed), "indicresearchkit_evaluation.csv",
            "text/csv", width="stretch",
        )
        right.download_button(
            "Download language summary CSV", csv_bytes(summary), "indicresearchkit_summary.csv",
            "text/csv", width="stretch",
        )
        st.markdown("**Translated datasets**")
        st.caption(
            "Each file contains text, the preserved label, and a 1-based source_row. "
            "Only successful forward translations are included."
        )
        for language in config["languages"]:
            translated = translated_dataset(raw_results, language)
            st.download_button(
                f"{language} · {len(translated)} translated rows",
                csv_bytes(translated),
                f"indicresearchkit_{LANGUAGES[language]}.csv",
                "text/csv",
                disabled=translated.empty,
            )


def main():
    st.set_page_config(page_title="IndicResearchKit", page_icon="🔬", layout="wide")
    st.caption("HINDI → INDIAN LANGUAGES → HINDI")
    st.title("IndicResearchKit")
    st.write("Explore multilingual prediction consistency with one fixed Hindi classifier.")

    env_key = dotenv_values(ROOT / ".env").get("SARVAM_API_KEY")
    api_key = (os.getenv("SARVAM_API_KEY") or env_key or "").strip()
    if api_key == "your_sarvam_api_key_here":
        api_key = ""

    with st.sidebar:
        st.header("Experiment setup")
        source = st.radio("Hindi dataset", ["Use sample dataset", "Upload CSV"])
        if source == "Upload CSV":
            uploaded = st.file_uploader("CSV with text and label columns", type=["csv"])
            content = uploaded.getvalue() if uploaded else None
        else:
            content = SAMPLE_PATH.read_bytes()
        st.download_button(
            "Download sample Hindi CSV", SAMPLE_PATH.read_bytes(), "sample_hindi.csv", "text/csv"
        )

    if content is None:
        st.info("Upload a UTF-8 Hindi CSV with text and label columns to begin.")
        return
    try:
        data = load_dataset(content)
    except ValueError as exc:
        st.error(str(exc))
        return

    fingerprint = hashlib.sha256(content).hexdigest()
    if st.session_state.get("dataset_fingerprint") != fingerprint:
        for key in ("trained_model", "results", "result_config"):
            st.session_state.pop(key, None)
        st.session_state["dataset_fingerprint"] = fingerprint

    with st.sidebar:
        languages = st.multiselect("Target languages", list(LANGUAGES), default=["Odia"])
        row_limit = st.number_input(
            "First N rows to translate", min_value=1, max_value=len(data),
            value=min(6, len(data)), step=1, key=f"row_limit_{fingerprint}",
            help="The classifier trains on the entire Hindi CSV. This limit only affects translation.",
        )
        st.caption(f"Up to {2 * row_limit * len(languages)} API requests, plus retries.")
        threshold = st.slider(
            "Review confidence threshold", min_value=0.0, max_value=1.0, value=0.60, step=0.05,
            help="Flag a sample if either model score is below this value. Changes do not call the API.",
        )
        if api_key:
            st.success("API key loaded")
        else:
            st.warning("Add SARVAM_API_KEY to .env to run translations.")
        st.caption("Sarvam Translate · sarvam-translate:v1")

    columns = st.columns(3)
    columns[0].metric("Hindi examples", len(data))
    columns[1].metric("Label classes", data["label"].nunique())
    columns[2].metric("Target languages", len(languages))
    with st.expander("Preview original Hindi dataset"):
        st.dataframe(data.head(10), hide_index=True, width="stretch")
        st.caption(
            "Showing up to 10 rows. The classifier trains on all original rows. "
            "The bundled sentiment data is a small, artificial demonstration."
        )

    st.subheader("Run a round-trip experiment")
    st.write(
        "Train on Hindi once, translate selected rows, then translate them back to Hindi "
        "and compare predictions from the unchanged model."
    )
    st.caption(
        "Selected text is sent to Sarvam's API. Labels stay local and are preserved in downloads. "
        "API usage may incur charges."
    )
    if not api_key:
        st.info("Copy .env.example to .env, set SARVAM_API_KEY, then rerun the app.")
    if not languages:
        st.info("Select at least one target language.")

    if st.button("Translate & evaluate", type="primary", disabled=not api_key or not languages):
        try:
            if "trained_model" not in st.session_state:
                with st.spinner("Training the classifier on the original Hindi dataset…"):
                    st.session_state["trained_model"] = train_classifier(data)
            progress = st.progress(0.0, text="Starting round-trip translations…")

            def report_progress(done, total):
                progress.progress(done / total, text=f"Processed {done} / {total} round trips")

            with SarvamTranslator(api_key) as translator:
                results = evaluate_dataset(
                    st.session_state["trained_model"], data.head(row_limit), languages,
                    translator, progress=report_progress,
                )
            st.session_state["results"] = results
            st.session_state["result_config"] = {
                "rows": row_limit, "languages": tuple(languages), "training_rows": len(data),
            }
            progress.empty()
        except (ValueError, TranslationError) as exc:
            st.error(str(exc))

    if "results" in st.session_state:
        config = st.session_state["result_config"]
        if (row_limit, tuple(languages)) != (config["rows"], config["languages"]):
            st.info("Results below use the previous run's rows and languages. Run again to use the current setup.")
        show_results(st.session_state["results"], config, threshold)


if __name__ == "__main__":
    main()
