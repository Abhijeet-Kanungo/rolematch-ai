from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.agent import RoleMatchAgent
from app.database import get_connection, init_db
from app.schemas import ChatRequest, SearchRequest
from app.search_engine import SemanticJobSearchEngine
from app.seed_jobs import seed_database

ROOT_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = ROOT_DIR / "static"

search_engine: Optional[SemanticJobSearchEngine] = None
career_agent: Optional[RoleMatchAgent] = None


class ProfileUpdateRequest(BaseModel):
    name: str
    skills: str
    preferred_location: str
    target_role: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize SQL database, load cached vector index, and boot LLM agent on startup."""
    global search_engine, career_agent
    init_db()

    conn = get_connection()
    job_count = conn.execute("SELECT COUNT(*) FROM jobs;").fetchone()[0]
    conn.close()
    if job_count == 0:
        seed_database()

    search_engine = SemanticJobSearchEngine()
    search_engine.build_or_load_index()
    career_agent = RoleMatchAgent(search_engine=search_engine)
    yield


app = FastAPI(
    title="RoleMatch AI — Semantic Job Search & Career Agent API",
    description="Hybrid SQL + Vector Semantic Job Search and Multi-Turn LLM Career Agent with Tool Calling and Pydantic Structured Outputs.",
    version="1.0.0",
    lifespan=lifespan,
)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def serve_frontend():
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="Frontend static/index.html not found.")
    return FileResponse(index_file)


@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "indexed_jobs": int(len(search_engine.job_ids)) if search_engine else 0,
        "active_model": career_agent.active_model if career_agent else None,
    }


@app.get("/api/stats")
def get_system_stats():
    """Live SQL analytics across companies, jobs, and search telemetry."""
    conn = get_connection()
    total_jobs = conn.execute("SELECT COUNT(*) FROM jobs;").fetchone()[0]
    total_companies = conn.execute("SELECT COUNT(*) FROM companies;").fetchone()[0]
    total_searches = conn.execute("SELECT COUNT(*) FROM search_logs;").fetchone()[0]
    avg_latency = conn.execute("SELECT ROUND(AVG(latency_ms), 1) FROM search_logs;").fetchone()[0] or 0.0
    conn.close()
    return {
        "total_jobs": total_jobs,
        "total_companies": total_companies,
        "total_searches": total_searches,
        "avg_search_latency_ms": avg_latency,
    }


@app.get("/api/jobs")
def list_jobs(
    location: Optional[str] = Query(default=None),
    work_mode: Optional[str] = Query(default=None),
    experience_level: Optional[str] = Query(default=None),
    limit: int = Query(default=20, ge=1, le=100),
):
    """Browse jobs using indexed SQL filters."""
    sql = """
        SELECT j.job_id, j.title, c.name AS company, c.industry,
               j.location, j.work_mode, j.experience_level,
               j.min_salary_lpa, j.max_salary_lpa, j.skills, j.description
        FROM jobs j
        INNER JOIN companies c ON j.company_id = c.company_id
        WHERE 1=1
    """
    params = []
    if location:
        sql += " AND LOWER(j.location) = LOWER(?)"
        params.append(location.strip())
    if work_mode:
        sql += " AND LOWER(j.work_mode) = LOWER(?)"
        params.append(work_mode.strip())
    if experience_level:
        sql += " AND LOWER(j.experience_level) = LOWER(?)"
        params.append(experience_level.strip())

    sql += " ORDER BY j.job_id ASC LIMIT ?"
    params.append(limit)

    conn = get_connection()
    rows = conn.execute(sql, params).fetchall()
    conn.close()
    return {"count": len(rows), "jobs": [dict(r) for r in rows]}


@app.post("/api/search")
def semantic_search(req: SearchRequest):
    """Execute hybrid SQL pre-filtering + cosine similarity ranking."""
    if not search_engine:
        raise HTTPException(status_code=503, detail="Search engine not initialized.")
    results = search_engine.search(
        query=req.query,
        location=req.location,
        work_mode=req.work_mode,
        experience_level=req.experience_level,
        min_salary_lpa=req.min_salary_lpa,
        top_k=req.top_k,
        user_id=req.user_id,
    )
    return {"query": req.query, "count": len(results), "results": results}


@app.post("/api/chat")
def chat_with_agent(req: ChatRequest):
    """Run multi-turn LLM career agent with tool calling and structured outputs."""
    if not career_agent:
        raise HTTPException(status_code=503, detail="Career agent not initialized.")
    try:
        return career_agent.chat(
            user_message=req.message,
            session_id=req.session_id,
            user_id=req.user_id,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Agent execution failed: {exc}")


@app.get("/api/profile/{user_id}")
def get_profile(user_id: int):
    conn = get_connection()
    row = conn.execute(
        "SELECT user_id, name, email, skills, preferred_location, target_role FROM users WHERE user_id = ?;",
        (user_id,),
    ).fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="User profile not found.")
    return dict(row)


@app.put("/api/profile/{user_id}")
def update_profile(user_id: int, req: ProfileUpdateRequest):
    conn = get_connection()
    cur = conn.execute(
        """
        UPDATE users
        SET name = ?, skills = ?, preferred_location = ?, target_role = ?
        WHERE user_id = ?;
        """,
        (req.name, req.skills, req.preferred_location, req.target_role, user_id),
    )
    conn.commit()
    conn.close()
    if cur.rowcount == 0:
        raise HTTPException(status_code=404, detail="User profile not found.")
    return {"status": "updated", "user_id": user_id}