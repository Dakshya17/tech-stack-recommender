#!/usr/bin/env python3
"""
recommender.py
---------------
Project 3: AI Recommendation Logic - Tech Stack Recommender.

Maps a user's raw skills to the job roles that best match them, using
content-based filtering: TF-IDF feature weighting + cosine similarity.

Pipeline (IPO model):
    INPUT   -> Collect at least 3 skills from the user
    PROCESS -> Vectorize the job role catalog and the user profile with
               TF-IDF, then score every role against the user profile
               using cosine similarity
    OUTPUT  -> Return the top-N highest-scoring roles, most relevant first

Run it with:
    python recommender.py
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from knowledge_base import load_dataset

TOP_N = 3
MIN_SKILLS_REQUIRED = 3


def build_corpus(job_roles: dict):
    """
    Turn the job role catalog into a text corpus TF-IDF can work with.

    Each role becomes one "document": its skills joined into a single
    string. Returns the role names (in a fixed order) and their matching
    documents, so scores can be mapped back to role names later.
    """
    role_names = list(job_roles.keys())
    documents = [",".join(job_roles[role]) for role in role_names]
    return role_names, documents


def build_vectorizer(documents):
    """
    Fit a TF-IDF vectorizer on the job role documents.

    TF-IDF down-weights skills that appear in almost every role (like
    "Git" or "Cloud Computing") and up-weights skills that are more
    distinctive, so common skills don't drown out a role's true signal.

    Documents are comma-separated skill lists, so the vectorizer is told
    to treat each comma-delimited entry as one token. This keeps
    multi-word skills like "Machine Learning" intact as a single feature
    instead of splitting into "machine" and "learning".
    """
    vectorizer = TfidfVectorizer(token_pattern=r"(?u)[^,]+", lowercase=True)
    role_vectors = vectorizer.fit_transform(documents)
    return vectorizer, role_vectors


def score_roles(user_skills: list, vectorizer, role_vectors, role_names):
    """
    Score every job role against the user's skill profile.

    The user's skills are joined into the same "document" format used for
    the roles, then transformed with the *already-fitted* vectorizer
    (skills the vectorizer has never seen are simply ignored - a known
    limitation discussed in the README under "Cold Start").

    Cosine similarity measures the angle between the user vector and each
    role vector: it captures how well the *pattern* of skills aligns,
    independent of how many skills the user listed.
    """
    user_document = ",".join(user_skills)
    user_vector = vectorizer.transform([user_document])

    similarities = cosine_similarity(user_vector, role_vectors)[0]

    scored = list(zip(role_names, similarities))
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored


def recommend(user_skills: list, top_n: int = TOP_N):
    """
    Full pipeline: build the corpus, vectorize it, score every role, and
    return the top-N recommended roles as (role_name, score) tuples.
    """
    job_roles = load_dataset()
    role_names, documents = build_corpus(job_roles)
    vectorizer, role_vectors = build_vectorizer(documents)
    scored = score_roles(user_skills, vectorizer, role_vectors, role_names)
    return scored[:top_n]


def parse_skills_input(raw_text: str) -> list:
    """Split comma-separated input into a clean list of skill strings."""
    return [skill.strip() for skill in raw_text.split(",") if skill.strip()]


def print_recommendations(recommendations, user_skills):
    print(f"\nBased on your skills ({', '.join(user_skills)}), here are your "
          f"top {len(recommendations)} recommended roles:\n")

    if all(score == 0 for _, score in recommendations):
        print("No overlap was found between your input and the known skill "
              "catalog. Try entering skills closer to standard tech terms "
              "(e.g. 'Python', 'AWS', 'React').")
        return

    for rank, (role, score) in enumerate(recommendations, start=1):
        print(f"{rank}. {role} - match score: {score:.2f} ({score:.0%})")


def main():
    print("=" * 60)
    print("  Project 3: AI Recommendation Logic - Tech Stack Recommender")
    print("=" * 60)
    print(f"\nEnter at least {MIN_SKILLS_REQUIRED} skills or interests, "
          "separated by commas.")
    print("Example: Python, Cloud Computing, Automation\n")

    while True:
        raw_text = input("Your skills: ")
        user_skills = parse_skills_input(raw_text)

        if len(user_skills) < MIN_SKILLS_REQUIRED:
            print(f"Please enter at least {MIN_SKILLS_REQUIRED} skills, "
                  f"separated by commas.")
            continue
        break

    recommendations = recommend(user_skills)
    print_recommendations(recommendations, user_skills)


if __name__ == "__main__":
    main()
