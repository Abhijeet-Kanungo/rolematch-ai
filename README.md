# RoleMatch AI — Semantic Job Search & LLM Career Agent

An end-to-end **AI Career Agent & Hybrid Semantic Job Search Engine** built with **FastAPI, 3NF SQL (SQLite), Gemini Vector Embeddings, Tool-Calling Agents, Pydantic Structured Outputs, and Docker**.

> **Live Demo:** `[PASTE_YOUR_RENDER_URL_HERE]`  
> **Interactive API Docs (Swagger UI):** `[PASTE_YOUR_RENDER_URL_HERE]/docs`

---

## Key Features
- **3NF Relational Database (`SQLite`):** Normalized schema (`companies`, `jobs`, `users`, `search_logs`) with foreign-key enforcement, `CHECK` constraints, indexed filter columns, and parameterized queries preventing SQL injection.
- **Hybrid Search Engine (SQL + Vectors):** Executes indexed SQL metadata pre-filtering (`location`, `work_mode`, `experience_level`, `min_salary_lpa`) followed by L2-normalized 768-dim cosine similarity ranking (`gemini-embedding-001`) over **520+ indexed jobs**.
- **Multi-Turn LLM Career Agent:** Explicit tool-calling loop (`get_user_profile`, `search_jobs`, `get_market_salary_stats`) with per-session conversation memory and strict **Pydantic** structured JSON synthesis to prevent hallucinated jobs.
- **Automated Retrieval & LLM Eval Suite (`eval.py`):** Quantitatively benchmarks Top-K retrieval accuracy, strict SQL filter compliance, tool selection accuracy, Pydantic schema validity, and SQL grounding rate.
- **Production Packaging:** Containerized with **Docker** and deployed live on the cloud with a single-server FastAPI + Tailwind web interface.

---

## System Architecture

```text
User (Web UI / Swagger / API)
        │
        ▼
  FastAPI Backend (app/main.py)
   ├── GET/PUT /api/profile/{id} ──► 3NF SQLite DB (companies, jobs, users, search_logs)
   ├── POST /api/search ───────────► Hybrid Engine: SQL Pre-Filter ──► Cosine Similarity (520 x 768)
   └── POST /api/chat ─────────────► Multi-Turn LLM Agent (app/agent.py)
                                       ├── Stage 1: Explicit Tool Loop (SQL + Vector Tools)
                                       └── Stage 2: Pydantic Structured JSON Synthesis
```

---

## Verified Evaluation Benchmarks (`eval_results.json`)

Measured across 8 hybrid retrieval queries and 3 end-to-end multi-tool agent scenarios via `python eval.py`:

| Metric | Result |
| :--- | :---: |
| **Top-3 Semantic Retrieval Accuracy** | **100.0%** (8/8) |
| **SQL Metadata Filter Compliance** | **100.0%** (8/8) |
| **Pydantic Structured Schema Pass Rate** | **100.0%** (3/3) |
| **Agent Tool Selection Accuracy** | **100.0%** (3/3) |
| **SQL Job Grounding Rate (Zero Hallucinations)** | **100.0%** (100% valid SQL IDs) |
| **Average Hybrid Search Latency** | **1,508.6 ms** |

---

## REST API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/health` | Service health check, indexed job count, active LLM model |
| `GET` | `/api/stats` | Live SQL analytics across jobs, companies, and search telemetry |
| `GET` | `/api/jobs` | Browse jobs with indexed SQL query filters |
| `POST` | `/api/search` | Hybrid SQL pre-filter + semantic vector search |
| `POST` | `/api/chat` | Multi-turn LLM career agent with tool-calling & structured JSON |
| `GET / PUT` | `/api/profile/{user_id}` | Read or update candidate profile in SQL `users` table |

---

## Run Locally

```bash
git clone [https://github.com/Abhijeet-Kanungo/rolematch-ai.git](https://github.com/Abhijeet-Kanungo/rolematch-ai.git)
cd rolematch-ai
python -m venv venv
venv\Scripts\activate          # macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file in the root folder:

```env
GEMINI_API_KEY=your_google_ai_studio_key
```

Seed the database and launch the server:

```bash
python -m app.seed_jobs
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000` (Web UI) or `http://127.0.0.1:8000/docs` (Swagger API docs).

### Run with Docker

```bash
docker build -t rolematch-ai .
docker run -p 8000:8000 -e GEMINI_API_KEY=your_key rolematch-ai
```