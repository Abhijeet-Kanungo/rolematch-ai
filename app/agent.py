import inspect
import json
import os
import time
from typing import Any, Optional

from dotenv import load_dotenv
from google import genai
from google.genai import types

from app.database import get_connection
from app.schemas import AgentChatResponse
from app.search_engine import SemanticJobSearchEngine

load_dotenv()

# Free-tier friendly model priority list (auto-falls back if one hits quota/404)
MODEL_CANDIDATES = [
    "gemini-3.1-flash-lite",
    "gemini-2.5-flash",
    "gemini-3.5-flash-lite",
]

SYSTEM_PROMPT = """You are RoleMatch AI, an expert career advisor and job-matching agent.
You have access to 3 live tools backed by a SQL database and vector search index:
1. get_user_profile: Fetch the candidate's saved skills, preferred location, and target role.
2. search_jobs: Run hybrid SQL + semantic vector search over 520+ indexed tech jobs.
3. get_market_salary_stats: Compute real SQL salary statistics (count, avg, min, max LPA) for a role keyword.

Rules:
- ALWAYS call tools to retrieve real data before recommending jobs or citing salaries. Never invent job IDs, companies, or salaries.
- If the user refers to "my profile", "my skills", or asks for personalized roles, call get_user_profile first, then call search_jobs using those skills.
- Keep your reasoning concise, accurate, and grounded strictly in the tool outputs.
"""


class RoleMatchAgent:
    """Multi-turn LLM career agent with tool calling, SQL grounding, and Pydantic structured outputs."""

    def __init__(self, search_engine: Optional[SemanticJobSearchEngine] = None) -> None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key or api_key == "YOUR_KEY_HERE":
            raise RuntimeError("GEMINI_API_KEY is missing in .env.")

        self.client = genai.Client(api_key=api_key)
        self.engine = search_engine or SemanticJobSearchEngine()
        if len(self.engine.job_ids) == 0:
            self.engine.build_or_load_index()

        # In-memory multi-turn conversation store: {session_id: [{"role": "user"|"model", "text": "..."}]}
        self.sessions: dict[str, list[dict[str, str]]] = {}
        self.active_model: str = MODEL_CANDIDATES[0]

    # ---------- SQL & Vector Tools Exposed to the LLM ----------

    def get_user_profile(self, user_id: int = 1) -> dict[str, Any]:
        """Fetch a candidate's profile (name, skills, preferred location, target role) from SQL."""
        conn = get_connection()
        row = conn.execute(
            "SELECT user_id, name, email, skills, preferred_location, target_role FROM users WHERE user_id = ?;",
            (int(user_id),),
        ).fetchone()
        conn.close()
        if not row:
            return {"error": f"No user found with user_id={user_id}"}
        return dict(row)

    def search_jobs(
        self,
        query: str,
        location: str = "",
        work_mode: str = "",
        experience_level: str = "",
        min_salary_lpa: float = 0.0,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Search 520+ jobs using hybrid SQL filters and semantic vector similarity.

        Args:
            query: Semantic description of desired role, tech stack, or responsibilities.
            location: Optional city filter (e.g., Bengaluru, Hyderabad, Pune, Gurugram, Remote). Leave empty if any.
            work_mode: Optional mode filter: Remote, Hybrid, or On-site. Leave empty if any.
            experience_level: Optional level filter: Intern, Entry-Level, or Mid-Level. Leave empty if any.
            min_salary_lpa: Optional minimum salary threshold in LPA (0 if no filter).
            top_k: Number of top matching jobs to return (1 to 10).
        """
        return self.engine.search(
            query=query,
            location=location.strip() or None,
            work_mode=work_mode.strip() or None,
            experience_level=experience_level.strip() or None,
            min_salary_lpa=float(min_salary_lpa) if min_salary_lpa and float(min_salary_lpa) > 0 else None,
            top_k=max(1, min(10, int(top_k))),
        )

    def get_market_salary_stats(self, role_keyword: str) -> dict[str, Any]:
        """Run a SQL aggregation query to compute salary benchmarks for a role keyword (e.g., 'AI', 'Data', 'Backend', 'SAP')."""
        conn = get_connection()
        pattern = f"%{role_keyword.strip()}%"
        rows = conn.execute(
            """
            SELECT experience_level,
                   COUNT(*) AS job_count,
                   ROUND(AVG(min_salary_lpa), 2) AS avg_min_lpa,
                   ROUND(AVG(max_salary_lpa), 2) AS avg_max_lpa,
                   ROUND(MAX(max_salary_lpa), 2) AS peak_lpa
            FROM jobs
            WHERE LOWER(title) LIKE LOWER(?) OR LOWER(skills) LIKE LOWER(?)
            GROUP BY experience_level
            ORDER BY avg_max_lpa DESC;
            """,
            (pattern, pattern),
        ).fetchall()
        conn.close()
        return {
            "role_keyword": role_keyword,
            "breakdown_by_level": [dict(r) for r in rows],
        }

    # ---------- Resilient Gemini Call Helper ----------

    def _generate_with_fallback(self, contents: Any, config: types.GenerateContentConfig) -> Any:
        """Try preferred models in order, handling 404/429 gracefully."""
        ordered_models = [self.active_model] + [m for m in MODEL_CANDIDATES if m != self.active_model]
        last_error: Optional[Exception] = None

        for model_name in ordered_models:
            for attempt in range(2):
                try:
                    resp = self.client.models.generate_content(
                        model=model_name,
                        contents=contents,
                        config=config,
                    )
                    self.active_model = model_name
                    return resp
                except Exception as exc:
                    last_error = exc
                    msg = str(exc)
                    if "404" in msg or "NOT_FOUND" in msg:
                        break  # Try next model immediately
                    if ("429" in msg or "RESOURCE_EXHAUSTED" in msg) and attempt == 0:
                        time.sleep(5)
                        continue
                    break

        raise RuntimeError(f"All Gemini models failed. Last error: {last_error}")

    # ---------- Main Multi-Turn Agent Execution ----------

    def chat(self, user_message: str, session_id: str = "default", user_id: int = 1) -> dict[str, Any]:
        """Execute Stage 1 (tool-calling loop) + Stage 2 (Pydantic structured output)."""
        start_time = time.perf_counter()
        history = self.sessions.setdefault(session_id, [])

        # Build conversation context (last 6 messages for token efficiency)
        contents: list[types.Content] = []
        for turn in history[-6:]:
            contents.append(
                types.Content(
                    role=turn["role"],
                    parts=[types.Part.from_text(text=turn["text"])],
                )
            )
        contents.append(
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=f"[Current user_id={user_id}] {user_message}")],
            )
        )

        # Register Python tools with manual execution control
        tool_registry = {
            "get_user_profile": self.get_user_profile,
            "search_jobs": self.search_jobs,
            "get_market_salary_stats": self.get_market_salary_stats,
        }
        tool_config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            tools=list(tool_registry.values()),
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            temperature=0.2,
        )

        tools_used: list[str] = []
        tool_traces: list[dict[str, Any]] = []
        draft_answer = ""

        # Check whether this SDK version supports passing function_call id to from_function_response
        supports_fr_id = "id" in inspect.signature(types.Part.from_function_response).parameters

        # STAGE 1: Explicit Agent Tool-Calling Loop (max 4 tool turns)
        for _ in range(4):
            response = self._generate_with_fallback(contents=contents, config=tool_config)
            function_calls = getattr(response, "function_calls", None)

            if not function_calls:
                draft_answer = response.text or ""
                break

            # Append the model's function-call turn to conversation
            if response.candidates and response.candidates[0].content:
                contents.append(response.candidates[0].content)

            tool_response_parts = []
            for fc in function_calls:
                fn_name = fc.name
                fn_args = dict(fc.args) if fc.args else {}
                tools_used.append(fn_name)

                try:
                    if fn_name not in tool_registry:
                        tool_result: Any = {"error": f"Unknown tool: {fn_name}"}
                    else:
                        tool_result = tool_registry[fn_name](**fn_args)
                except Exception as exc:
                    tool_result = {"error": str(exc)}

                tool_traces.append({"tool": fn_name, "args": fn_args, "output": tool_result})

                fr_kwargs: dict[str, Any] = {
                    "name": fn_name,
                    "response": {"result": tool_result},
                }
                if supports_fr_id and getattr(fc, "id", None):
                    fr_kwargs["id"] = fc.id
                tool_response_parts.append(types.Part.from_function_response(**fr_kwargs))

            contents.append(types.Content(role="user", parts=tool_response_parts))

        # STAGE 2: Strict Pydantic Structured JSON Output
        synthesis_prompt = f"""Format the grounded career assistant response into the exact required JSON schema.
Use ONLY real job_id, title, company, location, work_mode, salary_lpa, and match_percent values from the Tool Outputs below. Never invent jobs.

User Message: {user_message}

Recent Conversation History:
{json.dumps(history[-4:], indent=2)}

Tool Outputs (Ground Truth):
{json.dumps(tool_traces, indent=2)}

Agent Draft Reasoning:
{draft_answer}
"""
        structured_config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=AgentChatResponse,
            temperature=0.1,
        )
        structured_resp = self._generate_with_fallback(
            contents=synthesis_prompt,
            config=structured_config,
        )
        validated = AgentChatResponse.model_validate_json(structured_resp.text)

        # Save turn into session memory for multi-turn follow-ups
        history.append({"role": "user", "text": user_message})
        history.append({"role": "model", "text": validated.assistant_reply})

        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

        return {
            "session_id": session_id,
            "model_used": self.active_model,
            "tools_used": tools_used,
            "latency_ms": latency_ms,
            "response": validated.model_dump(),
        }


if __name__ == "__main__":
    agent = RoleMatchAgent()

    print("\n=== TURN 1: Personalized Tool-Calling Search ===")
    t1 = agent.chat(
        "Check my profile and recommend the top 3 Remote AI or LLM jobs that fit my skills.",
        session_id="demo_session",
        user_id=1,
    )
    print(f"Model: {t1['model_used']} | Tools: {t1['tools_used']} | Latency: {t1['latency_ms']} ms")
    print("Reply:", t1["response"]["assistant_reply"])
    for j in t1["response"]["recommended_jobs"]:
        print(f"  - [ID {j['job_id']} | {j['match_percent']}%] {j['title']} @ {j['company']} ({j['salary_lpa']})")
        print(f"    Why: {j['why_it_fits']} | Gaps: {j['skill_gaps']}")

    print("\n=== TURN 2: Multi-Turn Memory + SQL Salary Aggregation ===")
    t2 = agent.chat(
        "Now check the average market salary stats for AI roles in the database and tell me how those jobs compare.",
        session_id="demo_session",
        user_id=1,
    )
    print(f"Model: {t2['model_used']} | Tools: {t2['tools_used']} | Latency: {t2['latency_ms']} ms")
    print("Reply:", t2["response"]["assistant_reply"])
    print("Follow-ups:", t2["response"]["follow_up_questions"])