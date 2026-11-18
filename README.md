\# AI Data Analyst



A local AI-powered data analysis assistant built with Python, Streamlit, Pandas, and Ollama.



Ask questions about your datasets in natural language and receive analytical results, charts, and PDF reports.



\## Features



\- Natural language data analysis in Persian and English

\- Local LLM integration using Ollama

\- CSV and Excel dataset support

\- Automatic dataset profiling

\- Dataset schema detection

\- Business domain and metric detection

\- Natural language intent parsing

\- Data aggregation and analytical queries

\- Distinct value counting

\- Grouped business metrics

\- Chart generation

\- PDF report generation

\- Interactive Streamlit interface

\- Analysis history

\- Automated regression tests



\## Technology Stack



\- Python 3.11

\- Ollama

\- Pandas

\- NumPy

\- Matplotlib

\- Streamlit

\- ReportLab

\- Arabic Reshaper

\- Python Bidi

\- Pytest



\## Installation



```bash

git clone YOUR\_REPOSITORY\_URL

cd olama-project

python -m venv env

```



Windows PowerShell:



```powershell

.\\env\\Scripts\\Activate.ps1

pip install -r requirements.txt

```



\## Ollama Setup



```bash

ollama pull qwen2.5:3b

```



\## Configuration



Create a `.env` file:



```env

OLLAMA\_HOST=http://localhost:11434

OLLAMA\_MODEL=qwen2.5:3b

OUTPUT\_LANGUAGE=auto

```



\## Run Application



```bash

streamlit run app.py

```



\## Run Tests



```bash

python -m pytest -v

```



\## License



No license has been selected yet.

