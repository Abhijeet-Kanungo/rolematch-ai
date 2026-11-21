import random
from app.database import get_connection, init_db

COMPANIES = [
    ("Deloitte India", "Technology Consulting", "Hyderabad"),
    ("Elip AI", "AI Career Tech", "Gurugram"),
    ("Razorpay", "Fintech", "Bengaluru"),
    ("Zomato", "Consumer Tech", "Gurugram"),
    ("Swiggy", "Consumer Tech", "Bengaluru"),
    ("CRED", "Fintech", "Bengaluru"),
    ("PhonePe", "Fintech", "Bengaluru"),
    ("Meesho", "E-commerce", "Bengaluru"),
    ("Freshworks", "Enterprise SaaS", "Chennai"),
    ("Zoho", "Enterprise SaaS", "Chennai"),
    ("Atlassian India", "Developer Tools", "Bengaluru"),
    ("Microsoft India", "Cloud & Enterprise", "Hyderabad"),
    ("Amazon India", "Cloud & E-commerce", "Bengaluru"),
    ("SAP Labs India", "Enterprise Software", "Bengaluru"),
    ("Flipkart", "E-commerce", "Bengaluru"),
    ("Postman", "API Platform", "Bengaluru"),
    ("Groww", "Fintech", "Bengaluru"),
    ("Nykaa", "E-commerce", "Mumbai"),
    ("Ola Electric", "Mobility & AI", "Bengaluru"),
    ("Zerodha", "Fintech", "Bengaluru"),
    ("BrowserStack", "Cloud Testing SaaS", "Mumbai"),
    ("Chargebee", "SaaS", "Chennai"),
    ("Fractal Analytics", "Data & AI Consulting", "Mumbai"),
    ("Mu Sigma", "Decision Sciences", "Bengaluru"),
    ("TCS Digital", "IT Services", "Pune"),
    ("Infosys Specialist", "IT Services", "Pune"),
    ("HCLTech", "Enterprise Engineering", "Noida"),
    ("KPMG India", "Advisory & Tech", "Gurugram"),
    ("EY GDS", "Consulting & Analytics", "Bengaluru"),
    ("PwC India", "Technology Transformation", "Kolkata"),
]

ROLE_TEMPLATES = [
    {
        "titles": ["AI/ML Engineer", "Generative AI Engineer", "LLM Application Developer", "AI Agent Intern"],
        "skills": ["Python", "LLMs", "RAG", "Vector Embeddings", "FastAPI", "Prompt Engineering", "Docker", "SQL"],
        "focus": [
            "Build LLM-powered conversational agents with tool-calling, retrieval-augmented generation (RAG), and structured JSON outputs.",
            "Develop semantic search and embedding-based ranking pipelines over dynamic production datasets.",
            "Design automated LLM evaluation harnesses to measure retrieval precision, hallucination rate, and latency.",
        ],
    },
    {
        "titles": ["Backend Software Engineer", "Python Backend Developer", "SDE Intern - Backend", "API Platform Engineer"],
        "skills": ["Python", "FastAPI", "SQL", "PostgreSQL", "REST APIs", "Docker", "OOP", "DSA", "Git"],
        "focus": [
            "Design scalable REST APIs, relational SQL schemas, and reliable backend microservices in Python.",
            "Optimize database queries, indexing strategies, and API latency for high-traffic consumer workflows.",
            "Ship containerized backend services with automated unit tests, CI/CD, and cloud observability.",
        ],
    },
    {
        "titles": ["Data Analyst", "Business Intelligence Analyst", "Product Data Analyst", "Analytics Engineer"],
        "skills": ["SQL", "Python", "Pandas", "Power BI", "MS Excel", "ETL", "Data Visualization", "Statistics"],
        "focus": [
            "Perform exploratory data analysis (EDA) on large transaction datasets to uncover retention and revenue trends.",
            "Build interactive Power BI dashboards and automated SQL reporting pipelines for business stakeholders.",
            "Design clean ETL workflows to preprocess raw operational data into analytics-ready warehouse tables.",
        ],
    },
    {
        "titles": ["Data Engineer", "Cloud Data Engineer", "Junior Data Platform Engineer", "ETL Developer"],
        "skills": ["Python", "SQL", "DBMS", "ETL", "AWS", "Data Warehousing", "Docker", "Pandas"],
        "focus": [
            "Architect end-to-end data ingestion and transformation pipelines feeding enterprise analytics platforms.",
            "Maintain normalized relational schemas, data quality checks, and cloud-hosted data pipelines on AWS.",
            "Automate batch and near-real-time data workflows for cross-functional product and finance teams.",
        ],
    },
    {
        "titles": ["Technology & Transformation Analyst", "Enterprise Application Developer", "SAP ABAP Cloud Developer", "Cloud Solutions Analyst"],
        "skills": ["SAP ABAP Cloud", "Java", "SQL", "DBMS", "AWS", "OOP", "SDLC", "REST APIs"],
        "focus": [
            "Translate client business goals into cloud-enabled enterprise architectures and clean-core SAP extensions.",
            "Modernize legacy enterprise applications through API integration, cloud migration, and structured SDLC delivery.",
            "Collaborate with consulting teams to document requirements, design technical solutions, and run quality testing.",
        ],
    },
    {
        "titles": ["Data Scientist", "Applied Machine Learning Intern", "Decision Science Analyst", "Predictive Modeling Engineer"],
        "skills": ["Python", "Scikit-learn", "Pandas", "NumPy", "SQL", "Machine Learning", "Feature Engineering", "Matplotlib"],
        "focus": [
            "Develop supervised and unsupervised machine learning models for churn prediction, demand forecasting, and personalization.",
            "Run feature engineering, cross-validation, and A/B experiments to translate model metrics into business ROI.",
            "Productionize analytical Python pipelines replacing manual evaluation processes.",
        ],
    },
    {
        "titles": ["Java Backend Developer", "Software Engineer - Enterprise", "Associate Software Developer", "Full-Stack Analyst"],
        "skills": ["Java", "OOP", "DSA", "SQL", "DBMS", "Spring Boot", "REST APIs", "SDLC", "Git"],
        "focus": [
            "Develop modular object-oriented backend services in Java adhering to clean architecture and SDLC quality standards.",
            "Integrate enterprise relational databases, write complex SQL joins, and build reliable API layers.",
            "Troubleshoot production defects, optimize algorithmic efficiency, and contribute to agile sprint delivery.",
        ],
    },
]

LOCATIONS = ["Bengaluru", "Hyderabad", "Pune", "Gurugram", "Mumbai", "Chennai", "Noida", "Bhubaneswar", "Remote"]
WORK_MODES = ["Remote", "Hybrid", "On-site"]
EXP_LEVELS = [
    ("Intern", 3.6, 9.0),
    ("Entry-Level", 6.0, 16.0),
    ("Mid-Level", 14.0, 28.0),
]


def seed_database(target_jobs: int = 520) -> None:
    random.seed(42)
    init_db()
    conn = get_connection()
    cursor = conn.cursor()

    # Clear existing seed data so script can be re-run cleanly
    cursor.execute("DELETE FROM search_logs;")
    cursor.execute("DELETE FROM jobs;")
    cursor.execute("DELETE FROM users;")
    cursor.execute("DELETE FROM companies;")

    # 1. Insert Companies
    company_ids = []
    for name, industry, hq in COMPANIES:
        cursor.execute(
            "INSERT INTO companies (name, industry, hq_city) VALUES (?, ?, ?)",
            (name, industry, hq),
        )
        company_ids.append((cursor.lastrowid, name, industry))

    # 2. Insert 520 Diverse Job Listings
    for i in range(target_jobs):
        comp_id, comp_name, industry = random.choice(company_ids)
        template = random.choice(ROLE_TEMPLATES)
        title = random.choice(template["titles"])
        location = random.choice(LOCATIONS)
        work_mode = "Remote" if location == "Remote" else random.choice(WORK_MODES)
        exp_level, sal_low, sal_high = random.choice(EXP_LEVELS)

        min_sal = round(random.uniform(sal_low, (sal_low + sal_high) / 2), 1)
        max_sal = round(random.uniform(min_sal + 1.5, sal_high), 1)

        chosen_skills = random.sample(template["skills"], k=min(len(template["skills"]), random.randint(5, 7)))
        focus_sentence = random.choice(template["focus"])

        description = (
            f"{comp_name} ({industry}) is hiring a {exp_level} {title} in {location} ({work_mode}). "
            f"{focus_sentence} "
            f"Ideal candidates have strong fundamentals in {', '.join(chosen_skills[:4])} "
            f"and practical experience with {', '.join(chosen_skills[4:])}."
        )

        cursor.execute(
            """
            INSERT INTO jobs (
                company_id, title, location, work_mode, experience_level,
                min_salary_lpa, max_salary_lpa, skills, description
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                comp_id, title, location, work_mode, exp_level,
                min_sal, max_sal, ", ".join(chosen_skills), description
            ),
        )

    # 3. Insert Default Sample User Profile
    cursor.execute(
        """
        INSERT INTO users (name, email, skills, preferred_location, target_role)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            "Abhijeet Kanungo",
            "abhijeetkanungo00@gmail.com",
            "Python, Java, SQL, FastAPI, SAP ABAP Cloud, AWS, Power BI, LLMs, RAG",
            "Bengaluru",
            "AI/ML & Backend Software Engineer",
        ),
    )

    conn.commit()

    # Verification query using a SQL JOIN
    cursor.execute(
        """
        SELECT j.job_id, j.title, c.name AS company, j.location, j.work_mode, j.min_salary_lpa, j.max_salary_lpa
        FROM jobs j
        INNER JOIN companies c ON j.company_id = c.company_id
        LIMIT 5;
        """
    )
    sample_rows = cursor.fetchall()
    total_jobs = cursor.execute("SELECT COUNT(*) FROM jobs;").fetchone()[0]
    total_companies = cursor.execute("SELECT COUNT(*) FROM companies;").fetchone()[0]

    print(f"\nSeeded {total_companies} companies and {total_jobs} jobs successfully!")
    print("\nSample JOIN output (first 5 jobs):")
    for row in sample_rows:
        print(
            f"  [{row['job_id']}] {row['title']} @ {row['company']} "
            f"({row['location']} | {row['work_mode']} | {row['min_salary_lpa']}-{row['max_salary_lpa']} LPA)"
        )

    conn.close()


if __name__ == "__main__":
    seed_database()