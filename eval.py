import json
import time
from pathlib import Path

from app.agent import RoleMatchAgent
from app.database import get_connection
from app.schemas import AgentChatResponse
from app.search_engine import SemanticJobSearchEngine

RETRIEVAL_BENCHMARK = [
    {
        "query": "Build LLM agents with tool calling, RAG, and FastAPI",
        "filters": {"work_mode": "Remote"},
        "expected_keywords": ["llm", "ai", "rag", "agent"],
    },
    {
        "query": "SAP ABAP Cloud enterprise modernization and clean core",
        "filters": {"experience_level": "Entry-Level"},
        "expected_keywords": ["sap", "enterprise", "transformation", "cloud"],
    },
    {
        "query": "Power BI dashboards, SQL reporting, and exploratory data analysis",
        "filters": {},
        "expected_keywords": ["analyst", "power bi", "business intelligence", "analytics"],
    },
    {
        "query": "ETL data ingestion pipelines, AWS, and data warehousing",
        "filters": {"work_mode": "Hybrid"},
        "expected_keywords": ["data engineer", "etl", "aws", "warehousing"],
    },
    {
        "query": "Java Spring Boot microservices, OOP, and enterprise SQL",
        "filters": {"min_salary_lpa": 8.0},
        "expected_keywords": ["java", "spring", "software", "enterprise"],
    },
    {
        "query": "Scikit-learn predictive modeling, churn forecasting, feature engineering",
        "filters": {},
        "expected_keywords": ["data scientist", "machine learning", "predictive", "scikit-learn"],
    },
    {
        "query": "Python FastAPI REST APIs, PostgreSQL indexing, Docker backend",
        "filters": {"location": "Bengaluru"},
        "expected_keywords": ["backend", "api", "fastapi", "python"],
    },
    {
        "query": "Generative AI prompt engineering and LLM evaluation harness",
        "filters": {"experience_level": "Intern"},
        "expected_keywords": ["ai", "llm", "generative", "agent"],
    },
]

AGENT_BENCHMARK = [
    {
        "prompt": "Look up my profile and recommend 2 Remote AI or LLM jobs.",
        "expected_tools": ["get_user_profile", "search_jobs"],
    },
    {
        "prompt": "What are the average market salary benchmarks for Backend roles in the database?",
        "expected_tools": ["get_market_salary_stats"],
    },
    {
        "prompt": "Find 2 Entry-Level SAP or cloud transformation jobs in the database.",
        "expected_tools": ["search_jobs"],
    },
]


def run_eval() -> None:
    conn = get_connection()
    valid_job_ids = {int(r[0]) for r in conn.execute("SELECT job_id FROM jobs;").fetchall()}
    conn.close()

    engine = SemanticJobSearchEngine()
    engine.build_or_load_index()

    print("=== 1. Running Hybrid Retrieval Benchmark (8 Cases) ===")
    retrieval_hits = 0
    filter_passes = 0
    search_latencies = []

    for idx, case in enumerate(RETRIEVAL_BENCHMARK, start=1):
        t0 = time.perf_counter()
        results = engine.search(
            query=case["query"],
            top_k=3,
            log_search=False,
            **case["filters"],
        )
        ms = round((time.perf_counter() - t0) * 1000, 1)
        search_latencies.append(ms)

        # Check Top-3 relevance hit
        hit = any(
            any(kw in f"{r['title']} {r['skills']}".lower() for kw in case["expected_keywords"])
            for r in results
        )
        if hit:
            retrieval_hits += 1

        # Check strict SQL filter compliance
        f = case["filters"]
        compliant = bool(results) and all(
            (not f.get("location") or r["location"].lower() == f["location"].lower())
            and (not f.get("work_mode") or r["work_mode"].lower() == f["work_mode"].lower())
            and (not f.get("experience_level") or r["experience_level"].lower() == f["experience_level"].lower())
            and (
                not f.get("min_salary_lpa")
                or float(r["salary_lpa"].split("-")[1].split()[0]) >= float(f["min_salary_lpa"])
            )
            for r in results
        )
        if compliant:
            filter_passes += 1

        top_title = results[0]["title"] if results else "NONE"
        print(f"  Case {idx}: {'PASS' if hit else 'FAIL'} | Filter={'OK' if compliant else 'FAIL'} | {ms} ms | Top1: {top_title}")

    print("\n=== 2. Running End-to-End LLM Agent Benchmark (3 Cases) ===")
    agent = RoleMatchAgent(search_engine=engine)
    schema_passes = 0
    tool_passes = 0
    total_recs = 0
    grounded_recs = 0
    agent_latencies = []

    for idx, case in enumerate(AGENT_BENCHMARK, start=1):
        out = agent.chat(case["prompt"], session_id=f"eval_{idx}", user_id=1)
        agent_latencies.append(out["latency_ms"])

        # Schema validation check
        try:
            AgentChatResponse.model_validate(out["response"])
            schema_ok = True
            schema_passes += 1
        except Exception:
            schema_ok = False

        # Tool selection accuracy
        used = set(out["tools_used"])
        tool_ok = all(t in used for t in case["expected_tools"])
        if tool_ok:
            tool_passes += 1

        # SQL job_id grounding check (zero hallucination check)
        recs = out["response"].get("recommended_jobs", [])
        total_recs += len(recs)
        grounded_recs += sum(1 for r in recs if int(r["job_id"]) in valid_job_ids)

        print(
            f"  Agent {idx}: Schema={'OK' if schema_ok else 'FAIL'} | "
            f"Tools={out['tools_used']} ({'OK' if tool_ok else 'MISS'}) | "
            f"Recs={len(recs)} | {out['latency_ms']} ms"
        )
        time.sleep(2)

    summary = {
        "retrieval_cases": len(RETRIEVAL_BENCHMARK),
        "top3_retrieval_accuracy_pct": round(retrieval_hits / len(RETRIEVAL_BENCHMARK) * 100, 1),
        "sql_filter_compliance_pct": round(filter_passes / len(RETRIEVAL_BENCHMARK) * 100, 1),
        "avg_search_latency_ms": round(sum(search_latencies) / len(search_latencies), 1),
        "agent_cases": len(AGENT_BENCHMARK),
        "pydantic_schema_pass_rate_pct": round(schema_passes / len(AGENT_BENCHMARK) * 100, 1),
        "tool_selection_accuracy_pct": round(tool_passes / len(AGENT_BENCHMARK) * 100, 1),
        "sql_grounding_rate_pct": round((grounded_recs / total_recs * 100), 1) if total_recs else 100.0,
        "avg_agent_latency_ms": round(sum(agent_latencies) / len(agent_latencies), 1),
    }

    Path("eval_results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n================ EVAL SUMMARY ================")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print("Saved benchmark report to eval_results.json")


if __name__ == "__main__":
    run_eval()