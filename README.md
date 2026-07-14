# BeyondSearch

**Landing Screen For users to Ask Query
**
<img width="1912" height="907" alt="image" src="https://github.com/user-attachments/assets/014b5a0a-ae60-4d83-a937-a10a75f969cf" />

**Query is processed by the 3 Agents Sequentially 
**
<img width="1912" height="911" alt="image" src="https://github.com/user-attachments/assets/6ed7781f-e782-4d47-9ddc-a23920772f97" />

**Answer is displayed , Along with different routes of Tracing in Google Sheets, langsmith , Answer.txt. 
**
<img width="1920" height="2136" alt="screencapture-localhost-8501-2026-07-14-19_33_31" src="https://github.com/user-attachments/assets/2451bfb3-4d14-40d0-ace5-cb1b6461b6ab" />

Everything -- the 3-agent CrewAI pipeline, SQLite history, Google Sheets
logging, LangSmith tracing, and the Streamlit UI -- lives in one file:
`app.py`. No separate backend, no multi-file module structure.

Each stage runs in a background thread while the UI polls for live progress,
so the UI staying responsive never slows the agents down.

## Agents

1. **Knowledge Agent** -- answers from the model's own trained knowledge
   only. No tools, no web access. Fast; can be stale on time-sensitive facts.
2. **Web Research Agent** -- searches live via Serper.dev, writes an
   independent answer with inline `[n]` citations backed by real URLs.
3. **Recording Agent** -- no reasoning, just persists the result. Calls one
   tool (`Record Results`) that writes `answer.txt`, logs to Google Sheets,
   and saves to local history -- see below.

Sequential: Knowledge -> Research -> Recording, each stage's output feeding
the next.

## Recording: answer.txt vs. Google Sheets

- **`answer.txt`** -- always written, one file per question, saved to
  `data/answers/answer_<request_id>.txt`. Contains the question, both
  agents' answers, references, and timing. This is the primary record.
- **Google Sheets** -- extra, optional audit trail. Appends one row per
  query (timestamp, both answers, timings, status) to an `ExecutionLog`
  worksheet. Only runs if `GOOGLE_SHEETS_SPREADSHEET_ID` and the service
  account credentials are configured; if not, it's silently skipped -- no
  crash, nothing else breaks.

## LangSmith tracing

If `LANGCHAIN_API_KEY` is set, each query is wrapped in one root trace
("BeyondSearch - User Query") with a child span per LLM call and per tool
call (Web Search, Record Results) -- so prompts, responses, and latency for
every agent step are inspectable at smith.langchain.com. If not configured,
tracing is a no-op and the pipeline behaves identically either way.

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


