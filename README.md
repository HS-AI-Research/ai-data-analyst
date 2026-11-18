@'
# AI Data Analyst

A local AI-powered data analysis assistant built with Python, Streamlit, Pandas, and Ollama.

Analyze datasets using natural-language questions in Persian and English. Receive analytical results, exploratory insights, charts, and PDF reports.

## Features

- Persian and English natural-language analysis
- Local LLM integration using Ollama
- CSV and Excel support
- Automatic dataset profiling
- Exploratory Data Analysis (EDA)
- Intent classification
- Analytical query parsing
- Deterministic calculations using Pandas
- Aggregation, distinct counting, and grouped metrics
- Numeric, categorical, and date analysis
- Correlation and outlier detection
- Data-quality warnings
- Chart generation
- PDF report generation
- Streamlit interface
- Automated regression tests

## Architecture

The application supports two analysis paths:

1. Exploratory requests are routed through the Intent Classifier to the EDA Engine.
2. Specific analytical questions are parsed and executed by the Query Engine.

Pandas performs numerical calculations deterministically. Ollama is used for intent classification and natural-language interpretation when available. Fallback mechanisms support operation when the LLM is unavailable.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python 3.11 | Core programming language |
| Ollama | Local LLM |
| Pandas | Data processing |
| NumPy | Numerical computing |
| Matplotlib | Visualization |
| Streamlit | Web interface |
| ReportLab | PDF generation |
| Pytest | Automated testing |

## Project Structure

```text
ai-data-analyst/
├── app.py
├── requirements.txt
├── requirements-dev.txt
├── .env.example
├── .gitignore
├── README.md
├── data/
│   └── raw/
│       ├── orders.csv
│       ├── employees.csv
│       └── orders_test.xlsx
├── src/
│   ├── assistant.py
│   ├── intent_classifier.py
│   ├── intent_parser.py
│   ├── query_engine.py
│   ├── eda_engine.py
│   ├── metric_registry.py
│   ├── domain_registry.py
│   ├── schema_detector.py
│   ├── dataset_context.py
│   ├── data_loader.py
│   ├── data_profiler.py
│   ├── data_analyzer.py
│   ├── advanced_analyzer.py
│   ├── chart_generator.py
│   ├── report_generator.py
│   ├── ollama_client.py
│   ├── config.py
│   └── main.py
└── tests/
    ├── test_assistant.py
    ├── test_data_loader.py
    ├── test_eda_engine.py
    ├── test_intent_parser.py
    └── test_query_engine.py