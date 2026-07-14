# BeyondSearch

Everything -- the 3-agent CrewAI pipeline, SQLite history, Google Sheets
logging, LangSmith tracing, and the Streamlit UI -- lives in one file:
`app.py`. No separate backend, no multi-file module structure.

Pipeline: **Knowledge Agent** (answers from the model's own knowledge, no
web access) -> **Web Research Agent** (searches live via Serper.dev, writes an
independent cited answer) -> **Recording** (writes `answer.txt`, logs a row
to Google Sheets, saves the question/answers to local history). Each stage
runs in a background thread while the UI polls for live progress, so the UI
staying responsive never slows the agents down.

## 1. Setup (run once)

```bash
cd crewAgents
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

## 2. Configure credentials

`.env` already has your OpenAI key, model, and Serper.dev key wired in.

Still needed before these two come online (the app runs and degrades
gracefully without them -- it just skips Sheets logging / tracing):

- **Google Sheets**: `credentials/google_service_account.json` is already in
  place. You still need to:
  1. Create (or pick) a Google Sheet.
  2. Share it with Service Account user created via Google Console as **Editor**.
  3. Put that sheet's ID (the long string in its URL between `/d/` and `/edit`) into `GOOGLE_SHEETS_SPREADSHEET_ID` in `.env`.
- **LangSmith**: get a key at smith.langchain.com, set `LANGCHAIN_API_KEY` in `.env`.

## 3. Run

```bash
streamlit run app.py
```

Open the URL Streamlit prints (usually http://localhost:8501).

## 4. What to check on first run

- Ask a question -> the 3-stage timeline animates live (Knowledge -> Research -> Recording), then a single answer view renders: the Research Agent's cited answer as primary (or the Knowledge Agent's, if research failed/isn't configured), with the other agent's answer collapsed into an "Also answered by..." expander.
- If Google Sheets/LangSmith aren't configured, those show "not configured" in the sidebar, but the query still completes and `answer.txt` still downloads.
- Click "Download answer.txt" and confirm it has the question plus both agents' answers.
- Ask a follow-up question in the same session and check History in the sidebar.


