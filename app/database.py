import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "rolematch.db"


def get_connection() -> sqlite3.Connection:
    """Create a SQLite connection with Foreign Key enforcement enabled."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # Lets us access columns by name like a dict
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_db() -> None:
    """Create 3NF relational tables and indexes if they do not exist."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.executescript(
        """
        CREATE TABLE IF NOT EXISTS companies (
            company_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            industry TEXT NOT NULL,
            hq_city TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS jobs (
            job_id INTEGER PRIMARY KEY AUTOINCREMENT,
            company_id INTEGER NOT NULL,
            title TEXT NOT NULL,
            location TEXT NOT NULL,
            work_mode TEXT NOT NULL CHECK(work_mode IN ('Remote', 'Hybrid', 'On-site')),
            experience_level TEXT NOT NULL CHECK(experience_level IN ('Intern', 'Entry-Level', 'Mid-Level')),
            min_salary_lpa REAL NOT NULL,
            max_salary_lpa REAL NOT NULL,
            skills TEXT NOT NULL,
            description TEXT NOT NULL,
            posted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (company_id) REFERENCES companies(company_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            skills TEXT NOT NULL,
            preferred_location TEXT,
            target_role TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS search_logs (
            log_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            query_text TEXT NOT NULL,
            top_job_id INTEGER,
            latency_ms REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE SET NULL,
            FOREIGN KEY (top_job_id) REFERENCES jobs(job_id) ON DELETE SET NULL
        );

        CREATE INDEX IF NOT EXISTS idx_jobs_location ON jobs(location);
        CREATE INDEX IF NOT EXISTS idx_jobs_work_mode ON jobs(work_mode);
        CREATE INDEX IF NOT EXISTS idx_jobs_exp_level ON jobs(experience_level);
        CREATE INDEX IF NOT EXISTS idx_search_logs_user ON search_logs(user_id);
        """
    )

    conn.commit()
    conn.close()
    print(f"Database initialized successfully at: {DB_PATH}")


if __name__ == "__main__":
    init_db()