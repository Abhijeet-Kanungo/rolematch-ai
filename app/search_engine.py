import os
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types

from app.database import get_connection

load_dotenv()

EMBED_MODEL = "gemini-embedding-001"
EMBED_DIM = 768
CACHE_PATH = Path(__file__).resolve().parent.parent / "job_embeddings.npz"


class SemanticJobSearchEngine:
    """Hybrid search engine combining 3NF SQL filtering with vector cosine similarity."""

    def __init__(self) -> None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key or api_key == "YOUR_KEY_HERE":
            raise RuntimeError(
                "GEMINI_API_KEY is missing in .env! Paste your Google AI Studio key into .env first."
            )
        self.client = genai.Client(api_key=api_key)
        self.job_ids: np.ndarray = np.array([], dtype=np.int64)
        self.id_to_index: dict[int, int] = {}
        self.embeddings: np.ndarray = np.empty((0, EMBED_DIM), dtype=np.float32)

    @staticmethod
    def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
        """Normalize vectors to unit length so dot product == cosine similarity."""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.maximum(norms, 1e-12)

    def _embed_batch(self, texts: list[str], task_type: str) -> np.ndarray:
        """Embed a batch of texts using Gemini API with resilient 429 rate-limit backoff."""
        for attempt in range(5):
            try:
                response = self.client.models.embed_content(
                    model=EMBED_MODEL,
                    contents=texts,
                    config=types.EmbedContentConfig(
                        task_type=task_type,
                        output_dimensionality=EMBED_DIM,
                    ),
                )
                matrix = np.array(
                    [emb.values for emb in response.embeddings], dtype=np.float32
                )
                return self._l2_normalize(matrix)
            except Exception as exc:
                err_text = str(exc)
                if "429" in err_text or "RESOURCE_EXHAUSTED" in err_text:
                    wait_sec = 65
                    print(f"  Rate limit hit (attempt {attempt + 1}/5). Waiting {wait_sec}s...")
                    time.sleep(wait_sec)
                else:
                    raise
        raise RuntimeError("Embedding request failed after 5 retries.")

    def _save_cache(self) -> None:
        np.savez_compressed(CACHE_PATH, job_ids=self.job_ids, embeddings=self.embeddings)

    def build_or_load_index(self, force_rebuild: bool = False) -> None:
        """Load cached embeddings or incrementally embed missing SQL jobs with checkpointing."""
        conn = get_connection()
        rows = conn.execute(
            """
            SELECT j.job_id, j.title, c.name AS company, c.industry,
                   j.location, j.work_mode, j.experience_level, j.skills, j.description
            FROM jobs j
            INNER JOIN companies c ON j.company_id = c.company_id
            ORDER BY j.job_id ASC;
            """
        ).fetchall()
        conn.close()

        if not rows:
            raise RuntimeError("No jobs found in database. Run 'python -m app.seed_jobs' first.")

        existing_map: dict[int, np.ndarray] = {}
        if CACHE_PATH.exists() and not force_rebuild:
            data = np.load(CACHE_PATH)
            for jid, vec in zip(data["job_ids"], data["embeddings"]):
                existing_map[int(jid)] = vec

        # If every job in SQL is already cached, load immediately
        if len(existing_map) == len(rows) and all(int(r["job_id"]) in existing_map for r in rows):
            self.job_ids = np.array([int(r["job_id"]) for r in rows], dtype=np.int64)
            self.embeddings = np.vstack([existing_map[int(jid)] for jid in self.job_ids]).astype(np.float32)
            self.id_to_index = {int(jid): idx for idx, jid in enumerate(self.job_ids)}
            print(f"Loaded {len(self.job_ids)} cached job embeddings from {CACHE_PATH.name}")
            return

        # Find only jobs not yet embedded (supports safe resume)
        missing_rows = [r for r in rows if int(r["job_id"]) not in existing_map]
        print(
            f"Cached: {len(existing_map)}/{len(rows)}. "
            f"Embedding {len(missing_rows)} remaining jobs (80/min to respect Free Tier quota)..."
        )

        batch_size = 80
        for start in range(0, len(missing_rows), batch_size):
            batch_rows = missing_rows[start : start + batch_size]
            docs = [
                (
                    f"{r['title']} at {r['company']} ({r['industry']}). "
                    f"Level: {r['experience_level']} | Mode: {r['work_mode']} | City: {r['location']}. "
                    f"Skills: {r['skills']}. {r['description']}"
                )
                for r in batch_rows
            ]

            vecs = self._embed_batch(docs, task_type="RETRIEVAL_DOCUMENT")
            for r, vec in zip(batch_rows, vecs):
                existing_map[int(r["job_id"])] = vec

            # Save checkpoint after every batch
            sorted_ids = sorted(existing_map.keys())
            self.job_ids = np.array(sorted_ids, dtype=np.int64)
            self.embeddings = np.vstack([existing_map[jid] for jid in sorted_ids]).astype(np.float32)
            self._save_cache()

            done = len(existing_map)
            print(f"  Checkpoint saved: {done}/{len(rows)} jobs embedded.")

            if done < len(rows):
                print("  Pausing 62s for Free Tier rate-limit window...")
                time.sleep(62)

        self.id_to_index = {int(jid): idx for idx, jid in enumerate(self.job_ids)}
        print(f"Embedding index complete ({self.embeddings.shape}) -> {CACHE_PATH.name}")

    def search(
        self,
        query: str,
        location: Optional[str] = None,
        work_mode: Optional[str] = None,
        experience_level: Optional[str] = None,
        min_salary_lpa: Optional[float] = None,
        top_k: int = 5,
        user_id: Optional[int] = 1,
        log_search: bool = True,
    ) -> list[dict[str, Any]]:
        """Run hybrid search: SQL metadata pre-filtering + cosine similarity ranking."""
        if len(self.job_ids) == 0:
            self.build_or_load_index()

        start_time = time.perf_counter()

        # 1. SQL Pre-filtering (parameterized to prevent SQL injection)
        sql = """
            SELECT j.job_id, j.title, c.name AS company, c.industry,
                   j.location, j.work_mode, j.experience_level,
                   j.min_salary_lpa, j.max_salary_lpa, j.skills, j.description
            FROM jobs j
            INNER JOIN companies c ON j.company_id = c.company_id
            WHERE 1=1
        """
        params: list[Any] = []

        if location:
            sql += " AND LOWER(j.location) = LOWER(?)"
            params.append(location.strip())
        if work_mode:
            sql += " AND LOWER(j.work_mode) = LOWER(?)"
            params.append(work_mode.strip())
        if experience_level:
            sql += " AND LOWER(j.experience_level) = LOWER(?)"
            params.append(experience_level.strip())
        if min_salary_lpa is not None:
            sql += " AND j.max_salary_lpa >= ?"
            params.append(float(min_salary_lpa))

        conn = get_connection()
        candidates = conn.execute(sql, params).fetchall()

        if not candidates:
            conn.close()
            return []

        # 2. Vector Embedding of User Query
        query_vec = self._embed_batch([query], task_type="RETRIEVAL_QUERY")[0]

        # 3. Cosine Similarity via Fast Matrix Multiplication
        candidate_indices = [self.id_to_index[int(row["job_id"])] for row in candidates]
        candidate_matrix = self.embeddings[candidate_indices]
        similarities = candidate_matrix @ query_vec

        # 4. Sort and Select Top-K
        top_order = np.argsort(similarities)[::-1][:top_k]
        results: list[dict[str, Any]] = []
        for pos in top_order:
            row = candidates[int(pos)]
            score = float(similarities[int(pos)])
            results.append(
                {
                    "job_id": int(row["job_id"]),
                    "title": row["title"],
                    "company": row["company"],
                    "industry": row["industry"],
                    "location": row["location"],
                    "work_mode": row["work_mode"],
                    "experience_level": row["experience_level"],
                    "salary_lpa": f"{row['min_salary_lpa']}-{row['max_salary_lpa']} LPA",
                    "skills": row["skills"],
                    "description": row["description"],
                    "similarity_score": round(score, 4),
                    "match_percent": round(max(0.0, min(100.0, score * 100)), 1),
                }
            )

        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

        # 5. Persist Telemetry in SQL search_logs Table
        if log_search and results:
            conn.execute(
                """
                INSERT INTO search_logs (user_id, query_text, top_job_id, latency_ms)
                VALUES (?, ?, ?, ?)
                """,
                (user_id, query, results[0]["job_id"], latency_ms),
            )
            conn.commit()

        conn.close()
        return results


if __name__ == "__main__":
    engine = SemanticJobSearchEngine()
    engine.build_or_load_index()

    print("\n--- Test 1: Semantic AI Agent Search (Remote) ---")
    for job in engine.search(
        query="Build LLM chatbots with tool calling, RAG, and FastAPI in Python",
        work_mode="Remote",
        top_k=3,
    ):
        print(
            f"  [{job['match_percent']}%] {job['title']} @ {job['company']} "
            f"({job['location']} | {job['work_mode']} | {job['salary_lpa']})"
        )

    print("\n--- Test 2: Enterprise Cloud & SAP Search (Entry-Level) ---")
    for job in engine.search(
        query="Enterprise cloud transformation, SAP ABAP, Java, and SQL backend",
        experience_level="Entry-Level",
        top_k=3,
    ):
        print(
            f"  [{job['match_percent']}%] {job['title']} @ {job['company']} "
            f"({job['location']} | {job['work_mode']} | {job['salary_lpa']})"
        )