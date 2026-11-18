
"""
AI Data Analyst
===============

Streamlit application for local AI-powered data analysis.

Responsibilities:
- Manage Ollama connection and installed models.
- Activate and deactivate the application-level AI model.
- Load CSV and Excel datasets.
- Display dataset information and previews.
- Submit natural-language analytical questions.
- Execute deterministic and LLM-assisted analysis.
- Display answers, charts, and execution metadata.
- Maintain analysis history per dataset.
- Generate PDF reports on demand.
"""

import hashlib
import io
import re
import uuid
from pathlib import Path

import pandas as pd
import streamlit as st

from src.assistant import answer_question
from src.config import OLLAMA_MODEL
from src.data_loader import load_data
from src.ollama_client import check_ollama_connection
from src.report_generator import create_pdf_report


# =========================================================
# 1. APPLICATION CONFIGURATION
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent

DEFAULT_DATASET = (
    PROJECT_ROOT / "data" / "raw" / "orders.csv"
)

SUPPORTED_EXTENSIONS = ["csv", "xlsx", "xls"]


st.set_page_config(
    page_title="AI Data Analyst",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# 2. SESSION STATE
# =========================================================

def initialize_session_state():
    """
    Initialize application session state.

    Mutable objects are created independently for each session.
    """

    defaults = {
        "history": [],
        "dataset_id": None,

        "ollama_initialized": False,
        "ollama_connected": False,
        "installed_models": [],

        "active_model": None,
        "model_choice": None,

        "ollama_error": None,
        "model_warning": None,
    }

    for key, default_value in defaults.items():

        if key not in st.session_state:

            if isinstance(default_value, list):
                st.session_state[key] = []

            elif isinstance(default_value, dict):
                st.session_state[key] = {}

            else:
                st.session_state[key] = default_value


initialize_session_state()


# =========================================================
# 3. GENERAL HELPERS
# =========================================================

def detect_language(text):
    """
    Detect the language of a text.

    Returns:
        'fa' for Persian.
        'en' for English.
    """

    text = str(text)

    if re.search(r"[\u0600-\u06FF]", text):
        return "fa"

    return "en"


def resolve_project_path(file_path):
    """
    Resolve a relative path against the project directory.

    Absolute paths remain unchanged.
    """

    path = Path(file_path)

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return path.resolve()


def validate_chart_paths(raw_charts):
    """
    Validate chart paths returned by the assistant.

    Only existing files are returned.
    Duplicate paths are removed.
    """

    if not raw_charts:
        return []

    if isinstance(raw_charts, (str, Path)):
        raw_charts = [raw_charts]

    valid_charts = []
    seen_paths = set()

    for chart_path in raw_charts:

        try:

            path = resolve_project_path(chart_path)

            if not path.is_file():
                continue

            normalized_path = str(path)

            if normalized_path in seen_paths:
                continue

            seen_paths.add(normalized_path)
            valid_charts.append(normalized_path)

        except (TypeError, ValueError, OSError):
            continue

    return valid_charts


def get_dataset_identifier(uploaded_file, file_bytes):
    """
    Generate a dataset identifier based on file content.

    This detects replacement of a file even when its name
    and file size remain unchanged.
    """

    file_hash = hashlib.sha256(
        file_bytes
    ).hexdigest()

    return (
        f"upload:{uploaded_file.name}:{file_hash}"
    )


def load_uploaded_dataset(uploaded_file):
    """
    Load an uploaded CSV or Excel dataset.

    Returns:
        DataFrame, dataset identifier.
    """

    file_bytes = uploaded_file.getvalue()

    extension = Path(
        uploaded_file.name
    ).suffix.lower()

    buffer = io.BytesIO(file_bytes)

    if extension == ".csv":

        dataframe = pd.read_csv(buffer)

    elif extension in [".xlsx", ".xls"]:

        dataframe = pd.read_excel(buffer)

    else:

        raise ValueError(
            f"Unsupported file format: {extension}"
        )

    dataset_id = get_dataset_identifier(
        uploaded_file,
        file_bytes,
    )

    return dataframe, dataset_id


def load_default_dataset():
    """
    Load the default project dataset.

    Returns:
        DataFrame, dataset identifier.
    """

    if not DEFAULT_DATASET.is_file():
        return None, None

    dataframe = load_data(
        str(DEFAULT_DATASET)
    )

    stat = DEFAULT_DATASET.stat()

    dataset_id = (
        f"default:{stat.st_mtime_ns}:{stat.st_size}"
    )

    return dataframe, dataset_id


# =========================================================
# 4. OLLAMA MODEL MANAGER
# =========================================================

def refresh_ollama_status():
    """
    Refresh Ollama connection and installed model information.

    Model selection and model activation are separate operations.
    """

    try:

        status = check_ollama_connection()

        if not isinstance(status, dict):

            raise TypeError(
                "check_ollama_connection must return a dictionary."
            )

        connected = bool(
            status.get("connected", False)
        )

        models = status.get("models", []) or []

        models = sorted(
            {
                str(model).strip()
                for model in models
                if model is not None and str(model).strip()
            }
        )

        st.session_state.ollama_connected = connected
        st.session_state.installed_models = models

        st.session_state.ollama_error = (
            status.get("error")
        )

        # -------------------------------------------------
        # Connection unavailable
        # -------------------------------------------------

        if not connected:

            st.session_state.active_model = None
            st.session_state.model_choice = None

            st.session_state.model_warning = (
                "Ollama is not available. "
                "The active model has been cleared."
            )

            return

        # -------------------------------------------------
        # Validate currently active model
        # -------------------------------------------------

        active_model = st.session_state.active_model

        if active_model and active_model not in models:

            st.session_state.active_model = None

            st.session_state.model_warning = (
                f"Active model '{active_model}' is no longer installed. "
                "Please activate another model."
            )

        else:

            st.session_state.model_warning = None

        # -------------------------------------------------
        # Validate model selector
        # -------------------------------------------------

        current_choice = st.session_state.model_choice

        if current_choice not in models:

            if OLLAMA_MODEL in models:

                current_choice = OLLAMA_MODEL

            elif models:

                current_choice = models[0]

            else:

                current_choice = None

        st.session_state.model_choice = current_choice

    except Exception as error:

        st.session_state.ollama_connected = False
        st.session_state.installed_models = []

        st.session_state.active_model = None
        st.session_state.model_choice = None

        st.session_state.ollama_error = str(error)

        st.session_state.model_warning = (
            "Ollama connection failed. "
            "The active model has been cleared."
        )


def activate_model(model_name):
    """
    Activate an installed model for this application session.
    """

    if not st.session_state.ollama_connected:

        st.session_state.model_warning = (
            "Cannot activate a model because Ollama is disconnected."
        )

        return False

    if model_name not in st.session_state.installed_models:

        st.session_state.model_warning = (
            "The selected model is not installed. "
            "Refresh the model list."
        )

        return False

    st.session_state.active_model = model_name
    st.session_state.model_warning = None

    return True


def deactivate_model():
    """
    Deactivate the currently active application-level model.
    """

    st.session_state.active_model = None


# ---------------------------------------------------------
# Initial Ollama connection check
# ---------------------------------------------------------

if not st.session_state.ollama_initialized:

    refresh_ollama_status()

    st.session_state.ollama_initialized = True


# =========================================================
# 5. SIDEBAR: OLLAMA MODEL MANAGER
# =========================================================

st.sidebar.header("AI Model Manager")

# ---------------------------------------------------------
# Connection status
# ---------------------------------------------------------

if st.session_state.ollama_connected:

    st.sidebar.success(
        "Ollama Connected"
    )

else:

    st.sidebar.error(
        "Ollama Disconnected"
    )

    if st.session_state.ollama_error:

        st.sidebar.caption(
            str(st.session_state.ollama_error)
        )


# ---------------------------------------------------------
# Installed models
# ---------------------------------------------------------

st.sidebar.metric(
    "Installed Models",
    len(st.session_state.installed_models),
)


# ---------------------------------------------------------
# Active model
# ---------------------------------------------------------

if st.session_state.active_model:

    st.sidebar.success(
        f"Active Model: {st.session_state.active_model}"
    )

    if st.sidebar.button(
        "Deactivate",
        key="deactivate_model",
        use_container_width=True,
    ):

        deactivate_model()

        st.rerun()

else:

    st.sidebar.warning(
        "No active model"
    )


# ---------------------------------------------------------
# Model warnings
# ---------------------------------------------------------

if st.session_state.model_warning:

    st.sidebar.warning(
        st.session_state.model_warning
    )


# ---------------------------------------------------------
# Refresh installed models
# ---------------------------------------------------------

if st.sidebar.button(
    "Refresh Models",
    key="refresh_models",
    use_container_width=True,
):

    refresh_ollama_status()

    st.rerun()


# ---------------------------------------------------------
# Model selector
# ---------------------------------------------------------

installed_models = st.session_state.installed_models

if st.session_state.ollama_connected and installed_models:

    # Ensure the selector value exists in the current model list.
    if st.session_state.model_choice not in installed_models:

        st.session_state.model_choice = installed_models[0]

    selected_model = st.sidebar.selectbox(
        "Select a model",
        options=installed_models,
        key="model_choice",
    )

    if st.sidebar.button(
        "Activate",
        key="activate_model",
        type="primary",
        use_container_width=True,
    ):

        if activate_model(selected_model):

            st.rerun()

        else:

            st.sidebar.error(
                st.session_state.model_warning
            )

else:

    st.sidebar.info(
        "No installed models available. "
        "Check Ollama and refresh the model list."
    )


st.sidebar.caption(
    "The active model is used for LLM requests in this application. "
    "Model selection alone does not activate a model."
)

if st.session_state.active_model is None:

    st.sidebar.caption(
        "Deterministic analysis remains available without an active model."
    )


# =========================================================
# 6. SIDEBAR: DATA SOURCE
# =========================================================

st.sidebar.divider()

st.sidebar.header("Data Source")

uploaded_file = st.sidebar.file_uploader(
    "Upload your dataset",
    type=SUPPORTED_EXTENSIONS,
    key="dataset_uploader",
)

st.sidebar.caption(
    "Supported formats: CSV, XLSX, XLS"
)


# =========================================================
# 7. LOAD DATASET
# =========================================================

df = None
dataset_id = None

if uploaded_file is not None:

    try:

        df, dataset_id = load_uploaded_dataset(
            uploaded_file
        )

        st.sidebar.success(
            "Uploaded dataset loaded successfully."
        )

    except Exception as error:

        st.sidebar.error(
            f"Dataset loading failed: {error}"
        )

else:

    try:

        df, dataset_id = load_default_dataset()

        if df is not None:

            st.sidebar.success(
                "Default dataset loaded."
            )

        else:

            st.sidebar.info(
                "No default dataset found."
            )

    except Exception as error:

        st.sidebar.error(
            f"Default dataset loading failed: {error}"
        )


# ---------------------------------------------------------
# Validate loaded dataset
# ---------------------------------------------------------

if df is not None:

    if not isinstance(df, pd.DataFrame):

        st.error(
            "The loaded dataset is not a valid pandas DataFrame."
        )

        df = None
        dataset_id = None

    elif df.empty:

        st.warning(
            "The selected dataset contains no rows."
        )


# =========================================================
# 8. RESET HISTORY WHEN DATASET CHANGES
# =========================================================

if st.session_state.dataset_id != dataset_id:

    st.session_state.history = []

    st.session_state.dataset_id = dataset_id


# =========================================================
# 9. SIDEBAR: ANALYSIS HISTORY
# =========================================================

st.sidebar.divider()

st.sidebar.subheader(
    "Analysis History"
)

st.sidebar.metric(
    "Saved Analyses",
    len(st.session_state.history),
)

if st.sidebar.button(
    "Clear History",
    key="clear_history",
    use_container_width=True,
):

    st.session_state.history = []

    st.rerun()


# =========================================================
# 10. PAGE HEADER
# =========================================================

st.title(
    "📊 AI Data Analyst"
)

st.caption(
    "Local AI-powered Data Analysis Assistant"
)

st.divider()


# =========================================================
# 11. DATASET OVERVIEW
# =========================================================

if df is not None:

    st.subheader(
        "Dataset Overview"
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Rows",
        f"{df.shape[0]:,}",
    )

    col2.metric(
        "Columns",
        f"{df.shape[1]:,}",
    )

    col3.metric(
        "Missing Values",
        f"{int(df.isna().sum().sum()):,}",
    )

    with st.expander(
        "Preview Dataset",
        expanded=False,
    ):

        st.dataframe(
            df.head(20),
            width="stretch",
        )

        st.caption(
            f"Showing {min(len(df), 20):,} of {len(df):,} rows."
        )

    st.divider()


    # =====================================================
    # 12. QUESTION INPUT
    # =====================================================

    st.subheader(
        "Ask Your Data"
    )

    if st.session_state.active_model:

        st.caption(
            f"Active AI model: {st.session_state.active_model}"
        )

    else:

        st.info(
            "No AI model is active. "
            "Supported deterministic analyses can still run."
        )

    with st.form(
        "analysis_form",
        clear_on_submit=False,
    ):

        question = st.text_input(
            "Enter your question",
            placeholder=(
                "Example: What is the monthly sales trend?"
            ),
        )

        submitted = st.form_submit_button(
            "Analyze",
            type="primary",
            use_container_width=True,
        )


    # =====================================================
    # 13. RUN ANALYSIS
    # =====================================================

    if submitted:

        if not question or not question.strip():

            st.warning(
                "Please enter a question."
            )

        else:

            question = question.strip()

            with st.spinner(
                "Analyzing your data..."
            ):

                try:

                    # -------------------------------------
                    # Capture the active model
                    # -------------------------------------

                    active_model = (
                        st.session_state.active_model
                    )

                    # -------------------------------------
                    # Execute assistant
                    # -------------------------------------

                    analysis_result = answer_question(
                        df,
                        question,
                        model=active_model,
                    )

                    if not isinstance(analysis_result, dict):

                        raise TypeError(
                            "answer_question must return a dictionary."
                        )

                    # -------------------------------------
                    # Extract answer
                    # -------------------------------------

                    answer = analysis_result.get(
                        "answer",
                        "",
                    )

                    if answer is None:
                        answer = ""

                    answer = str(answer)

                    if not answer.strip():

                        answer = (
                            "The analysis completed, "
                            "but no textual answer was returned."
                        )

                    # -------------------------------------
                    # Extract charts
                    # -------------------------------------

                    raw_charts = analysis_result.get(
                        "charts",
                        [],
                    )

                    valid_charts = validate_chart_paths(
                        raw_charts
                    )

                    # -------------------------------------
                    # Extract model information
                    # -------------------------------------

                    result_model = analysis_result.get(
                        "selected_model"
                    )

                    if result_model is None:

                        result_model = analysis_result.get(
                            "model",
                            active_model,
                        )

                    # -------------------------------------
                    # Extract LLM execution information
                    # -------------------------------------

                    llm_used = bool(
                        analysis_result.get(
                            "llm_used",
                            False,
                        )
                    )

                    llm_attempted = bool(
                        analysis_result.get(
                            "llm_attempted",
                            llm_used,
                        )
                    )

                    llm_succeeded = bool(
                        analysis_result.get(
                            "llm_succeeded",
                            llm_used,
                        )
                    )

                    llm_error = analysis_result.get(
                        "llm_error"
                    )

                    # -------------------------------------
                    # Extract execution metadata
                    # -------------------------------------

                    execution_time = analysis_result.get(
                        "execution_time"
                    )

                    operation = analysis_result.get(
                        "operation"
                    )

                    # -------------------------------------
                    # Create history item
                    # -------------------------------------

                    history_item = {
                        "id": uuid.uuid4().hex,

                        "question": question,
                        "answer": answer,

                        "charts": valid_charts,

                        "model": result_model,

                        "llm_used": llm_used,
                        "llm_attempted": llm_attempted,
                        "llm_succeeded": llm_succeeded,
                        "llm_error": llm_error,

                        "execution_time": execution_time,
                        "operation": operation,

                        "pdf_data": None,
                        "pdf_filename": None,
                        "pdf_error": None,
                    }

                    # -------------------------------------
                    # Save result
                    # -------------------------------------

                    st.session_state.history.append(
                        history_item
                    )

                    st.success(
                        "Analysis completed successfully."
                    )

                except Exception as error:

                    st.error(
                        f"Analysis failed: {error}"
                    )


    # =====================================================
    # 14. ANALYSIS RESULTS
    # =====================================================

    st.divider()

    st.subheader(
        "Analysis Results"
    )

    if st.session_state.history:

        total_results = len(
            st.session_state.history
        )

        for index, result in enumerate(
            reversed(st.session_state.history)
        ):

            question = result.get(
                "question",
                "",
            )

            answer = result.get(
                "answer",
                "",
            )

            charts = result.get(
                "charts",
                [],
            )

            result_id = result.get(
                "id",
                str(index),
            )

            language = detect_language(
                question
            )

            with st.expander(
                (
                    f"Q{total_results - index}: "
                    f"{question}"
                ),
                expanded=(index == 0),
            ):

                # -----------------------------------------
                # Question
                # -----------------------------------------

                st.markdown(
                    "**Question:**"
                )

                st.write(
                    question
                )

                # -----------------------------------------
                # Answer
                # -----------------------------------------

                st.markdown(
                    "**Answer:**"
                )

                st.markdown(
                    answer
                )

                # -----------------------------------------
                # Execution information
                # -----------------------------------------

                st.divider()

                st.markdown(
                    "**Execution Information**"
                )

                info_col1, info_col2 = st.columns(2)

                model_name = result.get(
                    "model"
                )

                info_col1.write(
                    f"**Model:** {model_name or 'None'}"
                )

                llm_attempted = result.get(
                    "llm_attempted",
                    False,
                )

                llm_succeeded = result.get(
                    "llm_succeeded",
                    result.get("llm_used", False),
                )

                if llm_succeeded:

                    info_col2.success(
                        "LLM-assisted"
                    )

                elif llm_attempted:

                    info_col2.warning(
                        "LLM fallback failed"
                    )

                else:

                    info_col2.info(
                        "Deterministic"
                    )

                # -----------------------------------------
                # LLM error
                # -----------------------------------------

                llm_error = result.get(
                    "llm_error"
                )

                if llm_error:

                    st.warning(
                        f"LLM information: {llm_error}"
                    )

                # -----------------------------------------
                # Execution time
                # -----------------------------------------

                execution_time = result.get(
                    "execution_time"
                )

                if execution_time is not None:

                    st.caption(
                        f"Execution time: {execution_time}"
                    )

                # -----------------------------------------
                # Operation
                # -----------------------------------------

                operation = result.get(
                    "operation"
                )

                if operation:

                    st.caption(
                        f"Operation: {operation}"
                    )

                # -----------------------------------------
                # Generated charts
                # -----------------------------------------

                if charts:

                    st.divider()

                    st.markdown(
                        "**Generated Charts:**"
                    )

                    for chart_path in charts:

                        path = resolve_project_path(
                            chart_path
                        )

                        if path.is_file():

                            st.image(
                                str(path),
                                caption=path.name,
                                width="stretch",
                            )

                # -----------------------------------------
                # PDF report generation
                # -----------------------------------------

                st.divider()

                pdf_data = result.get(
                    "pdf_data"
                )

                pdf_filename = result.get(
                    "pdf_filename"
                )

                if pdf_data is None:

                    if st.button(
                        (
                            "ساخت گزارش PDF"
                            if language == "fa"
                            else "Generate PDF Report"
                        ),
                        key=f"generate_pdf_{result_id}",
                        use_container_width=True,
                    ):

                        try:

                            pdf_path = create_pdf_report(
                                question=question,
                                answer=answer,
                                chart_paths=charts,
                                language=language,
                            )

                            if not pdf_path:

                                raise ValueError(
                                    "PDF generator did not return a file path."
                                )

                            pdf_file_path = resolve_project_path(
                                pdf_path
                            )

                            if not pdf_file_path.is_file():

                                raise FileNotFoundError(
                                    f"PDF file not found: {pdf_file_path}"
                                )

                            with open(
                                pdf_file_path,
                                "rb",
                            ) as pdf_file:

                                pdf_data = pdf_file.read()

                            if not pdf_data:

                                raise ValueError(
                                    "Generated PDF file is empty."
                                )

                            result["pdf_data"] = pdf_data

                            result["pdf_filename"] = (
                                pdf_file_path.name
                            )

                            result["pdf_error"] = None

                            st.rerun()

                        except Exception as error:

                            result["pdf_error"] = str(
                                error
                            )

                            st.error(
                                f"PDF generation failed: {error}"
                            )

                else:

                    st.download_button(
                        label=(
                            "دریافت گزارش PDF"
                            if language == "fa"
                            else "Download PDF Report"
                        ),
                        data=pdf_data,
                        file_name=pdf_filename or "analysis_report.pdf",
                        mime="application/pdf",
                        key=f"download_pdf_{result_id}",
                        use_container_width=True,
                    )

                # -----------------------------------------
                # Previous PDF error
                # -----------------------------------------

                pdf_error = result.get(
                    "pdf_error"
                )

                if pdf_error:

                    st.error(
                        f"PDF generation failed: {pdf_error}"
                    )

    else:

        st.info(
            "No analysis results yet. "
            "Enter a question above to get started."
        )


# =========================================================
# 15. NO DATASET
# =========================================================

else:

    st.warning(
        "Please upload a dataset or check the default dataset."
    )

    st.info(
        "Upload a CSV or Excel file using the Data Source section "
        "in the sidebar."
    )


# =========================================================
# 16. FOOTER
# =========================================================

st.divider()

st.caption(
    "AI Data Analyst | Local Data Processing | "
    "Python · Streamlit · Pandas · Ollama"
)
