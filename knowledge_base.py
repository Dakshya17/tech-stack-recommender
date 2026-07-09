"""
knowledge_base.py
------------------
The domain knowledge layer for the Tech Stack Recommender.

This holds the "item" side of the content-based filtering system: a set of
job roles, each described by the skills, tools, and platforms typically
associated with it. The recommender engine (recommender.py) treats each
role as an item and matches it against a user's skill profile using
TF-IDF vectors and cosine similarity.

Keeping this data separate from the matching logic means the role catalog
can be extended or replaced (e.g. loaded from a CSV or database) without
touching the recommendation algorithm itself.
"""

# ---------------------------------------------------------------------------
# JOB ROLE CATALOG
# Each role maps to a list of skills/tools/platforms associated with it.
# These lists intentionally overlap (e.g. "Python" appears in several roles)
# so that TF-IDF weighting has something meaningful to do: skills that show
# up everywhere get down-weighted, while distinctive skills get emphasized.
# ---------------------------------------------------------------------------
JOB_ROLES = {
    "Data Scientist": [
        "Python", "SQL", "Machine Learning", "Statistics", "Pandas",
        "NumPy", "Data Visualization", "Jupyter", "Scikit-learn",
    ],
    "Machine Learning Engineer": [
        "Python", "Machine Learning", "TensorFlow", "PyTorch", "Docker",
        "MLOps", "Model Deployment", "Cloud Computing", "Data Structures",
    ],
    "DevOps Engineer": [
        "AWS", "Docker", "Kubernetes", "CI/CD", "Automation", "Linux",
        "Terraform", "Cloud Computing", "Git", "Monitoring",
    ],
    "Backend Developer": [
        "Java", "Python", "SQL", "APIs", "Data Structures", "Git",
        "Microservices", "Node.js", "Databases",
    ],
    "Frontend Developer": [
        "JavaScript", "React", "HTML", "CSS", "TypeScript", "Git",
        "UI Design", "Web Design", "Responsive Design",
    ],
    "Full Stack Developer": [
        "JavaScript", "React", "Node.js", "SQL", "APIs", "Git",
        "HTML", "CSS", "Databases",
    ],
    "Cloud Architect": [
        "AWS", "Cloud Computing", "Kubernetes", "Terraform", "Networking",
        "Security", "System Design", "Automation", "Docker",
    ],
    "Data Engineer": [
        "Python", "SQL", "ETL", "Data Pipelines", "Spark", "Airflow",
        "Cloud Computing", "Databases", "Data Structures",
    ],
    "Mobile Developer": [
        "Swift", "Kotlin", "Java", "Mobile UI", "APIs", "Git",
        "iOS", "Android", "Responsive Design",
    ],
    "QA Automation Engineer": [
        "Python", "Selenium", "Test Automation", "CI/CD", "Git",
        "Java", "API Testing", "Quality Assurance",
    ],
    "Security Engineer": [
        "Networking", "Security", "Linux", "Penetration Testing",
        "Cloud Computing", "Cryptography", "Automation", "Monitoring",
    ],
    "Database Administrator": [
        "SQL", "Databases", "Data Structures", "Performance Tuning",
        "Backup and Recovery", "Cloud Computing", "Security",
    ],
}


def load_dataset() -> dict:
    """Return the job role -> skills catalog."""
    return JOB_ROLES


def all_known_skills() -> list:
    """Return a sorted, de-duplicated list of every skill in the catalog."""
    skills = set()
    for skill_list in JOB_ROLES.values():
        skills.update(skill_list)
    return sorted(skills)


def catalog_summary() -> str:
    """Return a short, human-readable summary of the catalog."""
    lines = [
        f"Job roles: {len(JOB_ROLES)}",
        f"Unique skills tracked: {len(all_known_skills())}",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    # Quick manual check: python knowledge_base.py
    print(catalog_summary())
    print("\nKnown skills:")
    for skill in all_known_skills():
        print(f"  - {skill}")
