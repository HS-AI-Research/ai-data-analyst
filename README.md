# AI Data Analyst

A local AI-powered data analysis assistant built with **Python, Streamlit, Pandas, and Ollama**.

Ask questions about your datasets in natural language and receive analytical results, charts, and PDF reports — powered by a locally running Large Language Model (LLM).

## Features

- Natural language data analysis in Persian and English
- Local LLM integration using Ollama
- CSV and Excel dataset support
- Automatic dataset profiling
- Dataset schema detection
- Business domain and metric detection
- Natural language intent parsing
- Data aggregation and analytical queries
- Distinct value counting
- Grouped business metrics
- Chart generation
- PDF report generation
- Interactive Streamlit interface
- Analysis history
- Automated regression tests

## Technology Stack

| Technology | Purpose |
|---|---|
| Python 3.11 | Core programming language |
| Ollama | Local LLM execution |
| Pandas | Data processing and analysis |
| NumPy | Numerical computing |
| Matplotlib | Data visualization |
| Streamlit | Interactive web interface |
| ReportLab | PDF report generation |
| Arabic Reshaper | Persian text rendering |
| Python Bidi | Right-to-left text support |
| Pytest | Automated testing |

## Project Structure

```text
ai-data-analyst/
│
├── app.py
├── requirements.txt
├── requirements-dev.txt
├── .env.example
├── .gitignore
├── README.md
│
├── data/
│   └── raw/
│       ├── orders.csv
│       ├── employees.csv
│       └── orders_test.xlsx
│
├── src/
│   ├── advanced_analyzer.py
│   ├── assistant.py
│   ├── chart_generator.py
│   ├── config.py
│   ├── dataset_context.py
│   ├── data_analyzer.py
│   ├── data_loader.py
│   ├── data_profiler.py
│   ├── domain_registry.py
│   ├── intent_parser.py
│   ├── main.py
│   ├── metric_registry.py
│   ├── ollama_client.py
│   ├── query_engine.py
│   ├── report_generator.py
│   └── schema_detector.py
│
└── tests/
    ├── test_assistant.py
    ├── test_data_loader.py
    ├── test_intent_parser.py
    └── test_query_engine.py
```

## Requirements

- Python 3.11
- Ollama
- A locally available Ollama model
- Windows, Linux, or macOS

## Installation

Clone the repository:

```bash
git clone https://github.com/HS-AI-Research/ai-data-analyst.git
cd ai-data-analyst
```

Create a virtual environment:

```bash
python -m venv env
```

Activate the environment on Windows PowerShell:

```powershell
.\env\Scripts\Activate.ps1
```

Install the required dependencies:

```bash
python -m pip install -r requirements.txt
```

For development and testing:

```bash
python -m pip install -r requirements-dev.txt
```

## Ollama Setup

Install Ollama from:

https://ollama.com/

Download the default model:

```bash
ollama pull qwen2.5:3b
```

Make sure the Ollama service is running before starting the application.

## Configuration

Create a `.env` file in the project root.

You can use `.env.example` as a template.

Default configuration:

```env
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=qwen2.5:3b
OUTPUT_LANGUAGE=auto
```

Configuration options:

| Variable | Description | Default |
|---|---|---|
| `OLLAMA_HOST` | Ollama server address | `http://localhost:11434` |
| `OLLAMA_MODEL` | Selected Ollama model | `qwen2.5:3b` |
| `OUTPUT_LANGUAGE` | Response language | `auto` |

## Run Application

Start the Streamlit application:

```bash
streamlit run app.py
```

Open the local application in your browser using the URL displayed by Streamlit.

## Run Tests

Run the complete test suite:

```bash
python -m pytest -v
```

Check installed dependencies:

```bash
python -m pip check
```

## Sample Data

The repository includes sample datasets for testing and demonstration:

- `orders.csv`
- `employees.csv`
- `orders_test.xlsx`

You can also load your own supported CSV and Excel datasets.

## License

No open-source license has been selected yet.

Until a license is added to this repository, all rights are reserved by the copyright holder. Please contact the repository owner for permission to use, modify, or distribute this software.