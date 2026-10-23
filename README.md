# Branch 1 — RAG Resume and Job Description Ingestion

This branch implements ingestion and retrieval only. It does not yet run an interview, score answers, or use a voice pipeline.

## Structure

```text
G5-ai-interview-agent/
├── main.py
├── requirements.txt
├── .env.example
├── .gitignore
├── data/
│   ├── resume/
│   └── job_description/
└── interview_data/
    ├── chroma_db/
    ├── sessions/
    └── reports/
```

## Setup (Windows PowerShell)

```powershell
cd "C:\Users\onkar\OneDrive\Desktop\G5-ai-interview-agent"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Put your own resume in `data/resume/` and the JD in `data/job_description/`. Use the actual filenames in the command.

## Run tests

```powershell
python main.py --mode test
```

## Run ingestion + RAG retrieval

```powershell
python main.py --mode ingest --resume "data/resume/resume.pdf" --jd "data/job_description/job_description.txt"
```

The first run downloads the sentence-transformers embedding model, so it requires an internet connection. It stores a local Chroma index under `interview_data/chroma_db/`.

## Notes

- This branch does not need a Groq API key because it does not generate interview questions yet.
- Project title counting is heuristic. Inspect `project_titles` and adjust section parsing if your resume layout differs.
- Do not commit private resumes, generated indexes, `.env` files, or audio recordings.
- Treat document content as untrusted data, not instructions.
