from typing import Optional
from pydantic import BaseModel, Field


class JobRecommendation(BaseModel):
    job_id: int = Field(description="Exact job_id returned by the search_jobs tool.")
    title: str = Field(description="Exact job title from the database.")
    company: str = Field(description="Exact company name from the database.")
    location: str = Field(description="Job city or Remote.")
    work_mode: str = Field(description="Remote, Hybrid, or On-site.")
    salary_lpa: str = Field(description="Salary range string, e.g. '10.5-15.2 LPA'.")
    match_percent: float = Field(description="Semantic match percentage from search_jobs.")
    why_it_fits: str = Field(description="1-2 concise sentences explaining why this role fits the user.")
    skill_gaps: list[str] = Field(description="1-3 skills listed in the job that the user should strengthen.")


class AgentChatResponse(BaseModel):
    assistant_reply: str = Field(
        description="Helpful, grounded conversational response directly answering the user's message."
    )
    recommended_jobs: list[JobRecommendation] = Field(
        description="Ranked list of jobs retrieved from search_jobs (empty list if no jobs were searched)."
    )
    follow_up_questions: list[str] = Field(
        description="2-3 short, actionable next questions the user might want to ask."
    )


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="User's message to the career agent.")
    session_id: str = Field(default="default", description="Session identifier for multi-turn memory.")
    user_id: int = Field(default=1, description="SQL user_id for personalized recommendations.")


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    location: Optional[str] = None
    work_mode: Optional[str] = None
    experience_level: Optional[str] = None
    min_salary_lpa: Optional[float] = None
    top_k: int = Field(default=5, ge=1, le=20)
    user_id: int = Field(default=1)