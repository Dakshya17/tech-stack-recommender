# Tech Stack Recommender

**Project 3** of the DecodeLabs AI Industrial Training Kit — the personalization
phase. This moves beyond classification (Project 2) into recommendation logic:
matching a user's profile against a catalog of items using similarity, not
just predicting a single label.

## What This Project Does

Takes a list of a user's skills or interests and recommends the job roles
that best match them, using **content-based filtering**:

1. **Ingestion** — the user provides at least 3 skills (e.g. `Python, Cloud Computing, Automation`)
2. **Feature extraction (TF-IDF)** — every job role in the catalog, and the
   user's own skill list, are converted into weighted numeric vectors
3. **Scoring (cosine similarity)** — the user's vector is compared against
   every role's vector to measure how closely their *pattern* of skills aligns
4. **Sorting and filtering** — roles are ranked by similarity score, and only
   the top matches are shown, so the user isn't overwhelmed with the full list

## Project Structure

```
tech-stack-recommender/
├── knowledge_base.py   # The item catalog: job roles and their associated skills
├── recommender.py       # The recommendation engine: TF-IDF + cosine similarity
├── requirements.txt
└── README.md
```

### Why split `knowledge_base.py` from `recommender.py`?

- `knowledge_base.py` — the domain data: which skills belong to which job
  role. This is what would change if you pointed the system at a different
  catalog (e.g. movies, products, courses) instead of tech roles.
- `recommender.py` — the algorithm: how any user profile gets scored against
  any catalog. This logic doesn't care what the items represent.

## Getting Started

### Prerequisites
- Python 3.7+

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run it

```bash
python recommender.py
```

### Example session

```
============================================================
  Project 3: AI Recommendation Logic - Tech Stack Recommender
============================================================

Enter at least 3 skills or interests, separated by commas.
Example: Python, Cloud Computing, Automation

Your skills: Python, Machine Learning, Statistics

Based on your skills (Python, Machine Learning, Statistics), here are your
top 3 recommended roles:

1. Data Scientist - match score: 0.53 (53%)
2. Machine Learning Engineer - match score: 0.30 (30%)
3. Backend Developer - match score: 0.12 (12%)
```

You can also inspect the role catalog on its own:

```bash
python knowledge_base.py
```

## How It Works

### Why content-based filtering?

Recommendation systems generally fall into two approaches:

| Approach | Driven by | Needs |
|---|---|---|
| Collaborative filtering | patterns across many users ("people who liked X also liked Y") | large historical interaction data |
| Content-based filtering | attributes of the items themselves | just a catalog of item features |

This project uses **content-based filtering** because it works immediately
with zero historical data — a hard requirement when there's no existing
user base to learn from yet.

### Why TF-IDF instead of simple keyword overlap?

The simplest similarity check just counts overlapping skills. But that
treats a generic skill like "Cloud Computing" (which appears in half the
catalog) exactly the same as a distinctive one like "Terraform." TF-IDF
(Term Frequency - Inverse Document Frequency) fixes this by:

- **Term Frequency** — how often a skill appears within one role's profile
- **Inverse Document Frequency** — a penalty for skills that appear across
  many roles, since they carry less discriminating power

The result: rare, specific skills get more weight in the similarity
calculation than common ones.

### Why cosine similarity instead of Euclidean distance?

Euclidean distance measures the straight-line distance between two vectors
and is sensitive to their magnitude — a role with a longer skill list would
appear "farther away" even if its skills perfectly match the user's
interests. Cosine similarity instead measures the *angle* between two
vectors, so it captures alignment of interests regardless of how many
skills either side lists. Scores range from 0 (no shared pattern) to 1
(perfectly aligned).

### The Cold Start Problem

Both this project and recommendation systems in general share a known
limitation: if a user's input has zero overlap with the known catalog (for
example, entering skills that were never seen during training), there is no
signal to compute a meaningful score from. This project handles that
case explicitly — see `print_recommendations()` in `recommender.py` — by
detecting an all-zero result and returning a clear message instead of a
misleading "top 3" list. In production systems, this is usually solved
with onboarding surveys, trending/popularity fallbacks, or metadata
inference.

## Extending the Project

- Add more job roles or skills to `JOB_ROLES` in `knowledge_base.py`
- Load the catalog from a CSV file instead of a hardcoded dictionary
- Let the user rate how important each skill is, and weight the input
  vector accordingly instead of treating all skills equally
- Add a `--top` command-line flag to control how many results are shown
- Swap in `scikit-learn`'s `CountVectorizer` to compare a binary/Jaccard
  approach against the TF-IDF approach and see how the rankings differ

## Roadmap

This project uses **content-based filtering**, which only looks at item
attributes. A natural next step is **collaborative filtering**, which
incorporates patterns across many users' behavior — the approach behind
recommendation engines like Netflix's and Amazon's — but that requires
substantial historical interaction data that this project intentionally
avoids needing.

## License

This project is licensed under the MIT License.

## Credits

Built as part of the DecodeLabs AI Industrial Training Kit — Batch 2026.
