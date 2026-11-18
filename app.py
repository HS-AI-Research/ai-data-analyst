
import streamlit as st
import pandas as pd
import os
import re

from src.data_loader import load_data
from src.assistant import answer_question
from src.report_generator import create_pdf_report


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="AI Data Analyst",
    page_icon="📊",
    layout="wide"
)


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "history" not in st.session_state:
    st.session_state.history = []

if "dataset_id" not in st.session_state:
    st.session_state.dataset_id = None


# --------------------------------------------------
# LANGUAGE DETECTION
# --------------------------------------------------

def detect_language(text):
    """Detect Persian or English based on text characters."""

    if re.search(r"[\u0600-\u06FF]", str(text)):
        return "fa"

    return "en"


# --------------------------------------------------
# PAGE HEADER
# --------------------------------------------------

st.title("📊 AI Data Analyst")
st.caption("Local AI-powered Data Analysis Assistant")

st.divider()


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.header("Data Source")

uploaded_file = st.sidebar.file_uploader(
    "Upload your dataset",
    type=["csv", "xlsx", "xls"]
)

st.sidebar.info(
    "Supported formats: CSV and Excel"
)


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

df = None
dataset_id = None

if uploaded_file is not None:

    try:

        if uploaded_file.name.lower().endswith(".csv"):

            df = pd.read_csv(uploaded_file)

        else:

            df = pd.read_excel(uploaded_file)

        dataset_id = (
            f"upload:{uploaded_file.name}:{uploaded_file.size}"
        )

        st.sidebar.success(
            "Dataset loaded successfully"
        )

    except Exception as error:

        st.sidebar.error(
            f"Loading error: {error}"
        )

else:

    default_file = "data/raw/orders.csv"

    if os.path.exists(default_file):

        try:

            df = load_data(default_file)

            dataset_id = (
                f"default:{os.path.getmtime(default_file)}"
            )

            st.sidebar.success(
                "Default dataset loaded"
            )

        except Exception as error:

            st.sidebar.error(
                f"Loading error: {error}"
            )


# --------------------------------------------------
# RESET HISTORY WHEN DATASET CHANGES
# --------------------------------------------------

if df is not None:

    if st.session_state.dataset_id != dataset_id:

        st.session_state.history = []
        st.session_state.dataset_id = dataset_id


# --------------------------------------------------
# HISTORY CONTROLS
# --------------------------------------------------

st.sidebar.divider()

st.sidebar.subheader(
    "Analysis History"
)

st.sidebar.metric(
    "Saved Analyses",
    len(st.session_state.history)
)

if st.sidebar.button("Clear History"):

    st.session_state.history = []

    st.rerun()


# --------------------------------------------------
# MAIN APPLICATION
# --------------------------------------------------

if df is not None:

    # ----------------------------------------------
    # DATASET OVERVIEW
    # ----------------------------------------------

    st.subheader(
        "Dataset Overview"
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Rows",
        df.shape[0]
    )

    col2.metric(
        "Columns",
        df.shape[1]
    )

    col3.metric(
        "Missing Values",
        int(df.isna().sum().sum())
    )

    with st.expander("Preview Dataset"):

        st.dataframe(
            df.head(20),
            width="stretch"
        )

    st.divider()

    # ----------------------------------------------
    # QUESTION INPUT
    # ----------------------------------------------

    st.subheader(
        "Ask Your Data"
    )

    with st.form("analysis_form"):

        question = st.text_input(
            "Enter your question",
            placeholder=(
                "Example: What is the monthly sales trend?"
            )
        )

        submitted = st.form_submit_button(
            "Analyze",
            type="primary"
        )

    # ----------------------------------------------
    # RUN ANALYSIS
    # ----------------------------------------------

    if submitted:

        if not question.strip():

            st.warning(
                "Please enter a question."
            )

        else:

            with st.spinner(
                "Analyzing your data..."
            ):

                try:

                    # Run data analysis
                    answer = answer_question(
                        df,
                        question
                    )

                    # ----------------------------------
                    # EXTRACT CHART PATHS
                    # ----------------------------------

                    chart_paths = re.findall(
                        r"outputs[\\/]+charts[\\/]+[^\s`]+\.png",
                        answer,
                        flags=re.IGNORECASE
                    )

                    # Keep only existing chart files
                    valid_charts = []

                    for chart_path in chart_paths:

                        normalized_path = os.path.normpath(
                            chart_path
                        )

                        if (
                            os.path.exists(normalized_path)
                            and normalized_path not in valid_charts
                        ):

                            valid_charts.append(
                                normalized_path
                            )

                    # ----------------------------------
                    # SAVE ANALYSIS RESULT
                    # ----------------------------------

                    result = {
                        "question": question,
                        "answer": answer,
                        "charts": valid_charts
                    }

                    st.session_state.history.append(
                        result
                    )

                except Exception as error:

                    st.error(
                        f"Analysis failed: {error}"
                    )

    # ----------------------------------------------
    # DISPLAY ANALYSIS HISTORY
    # ----------------------------------------------

    st.divider()

    st.subheader(
        "Analysis Results"
    )

    if st.session_state.history:

        for index, result in enumerate(
            reversed(st.session_state.history)
        ):

            question = result["question"]
            answer = result["answer"]
            charts = result["charts"]

            language = detect_language(
                question
            )

            with st.expander(
                f"Q{len(st.session_state.history) - index}: "
                f"{question}",
                expanded=(index == 0)
            ):

                # ----------------------------------
                # QUESTION
                # ----------------------------------

                st.markdown(
                    "**Question:**"
                )

                st.write(
                    question
                )

                # ----------------------------------
                # ANSWER
                # ----------------------------------

                st.markdown(
                    "**Answer:**"
                )

                st.markdown(
                    answer
                )

                # ----------------------------------
                # DISPLAY CHARTS
                # ----------------------------------

                if charts:

                    st.markdown(
                        "**Generated Charts:**"
                    )

                    for chart_path in charts:

                        if os.path.exists(chart_path):

                            st.image(
                                chart_path,
                                caption=os.path.basename(
                                    chart_path
                                ),
                                width="stretch"
                            )

                # ----------------------------------
                # PDF REPORT DOWNLOAD
                # ----------------------------------

                st.divider()

                try:

                    pdf_path = create_pdf_report(
                        question=question,
                        answer=answer,
                        chart_paths=charts,
                        language=language
                    )

                    with open(
                        pdf_path,
                        "rb"
                    ) as pdf_file:

                        pdf_data = pdf_file.read()

                    download_label = (
                        "دریافت گزارش PDF"
                        if language == "fa"
                        else "Download PDF Report"
                    )

                    st.download_button(
                        label=download_label,
                        data=pdf_data,
                        file_name=os.path.basename(
                            pdf_path
                        ),
                        mime="application/pdf",
                        key=f"download_pdf_{index}"
                    )

                except Exception as error:

                    st.error(
                        f"PDF generation failed: {error}"
                    )

    else:

        st.info(
            "No analysis results yet. "
            "Enter a question above to get started."
        )

else:

    st.warning(
        "Please upload a dataset or check the default dataset."
    )
