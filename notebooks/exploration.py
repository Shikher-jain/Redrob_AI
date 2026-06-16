"""
notebooks/exploration.py
========================
Interactive exploration of the candidate pool and scoring logic.
Run cell-by-cell in Jupyter or as a script.
Convert to .ipynb with: jupytext --to notebook exploration.py
"""

# %% [markdown]
# # Redrob Hackathon — Candidate Pool Exploration

# %% Imports
import sys, json
sys.path.insert(0, "../src")

import numpy as np
import pandas as pd
from pathlib import Path

# %% Load sample candidates
with open("../data/sample_candidates.json") as f:
    candidates = json.load(f)

print(f"Candidates: {len(candidates)}")

# %% Overview table
rows = []
for c in candidates:
    p   = c["profile"]
    sig = c.get("redrob_signals", {})
    rows.append({
        "id":        c["candidate_id"],
        "title":     p["current_title"],
        "yoe":       p["years_of_experience"],
        "location":  p["location"],
        "country":   p["country"],
        "otw":       sig.get("open_to_work_flag"),
        "notice":    sig.get("notice_period_days"),
        "resp_rate": sig.get("recruiter_response_rate"),
        "last_active": sig.get("last_active_date"),
        "n_skills":  len(c.get("skills", [])),
    })

df = pd.DataFrame(rows)
print(df.to_string())

# %% Score distribution
from feature_engineering import extract_all_features

features = []
for c in candidates:
    f = extract_all_features(c)
    f["candidate_id"] = c["candidate_id"]
    f["title"] = c["profile"]["current_title"]
    features.append(f)

fdf = pd.DataFrame(features).set_index("candidate_id")
print("\nFeature statistics:")
print(fdf.drop(columns=["title"]).describe().round(3))

# %% Top candidates by career quality
print("\nTop 10 by career_quality_score:")
print(fdf.nlargest(10, "career_quality_score")[
    ["title", "career_quality_score", "skill_match_score", "behavioral_engagement_score"]
].to_string())

# %% Skill distribution
all_skills = []
for c in candidates:
    for s in c.get("skills", []):
        all_skills.append({
            "name": s["name"],
            "proficiency": s["proficiency"],
            "endorsements": s["endorsements"],
            "duration_months": s.get("duration_months", 0),
        })

sdf = pd.DataFrame(all_skills)
print("\nTop 20 skills by frequency:")
print(sdf["name"].value_counts().head(20))

# %% Honeypot analysis
from feature_engineering import honeypot_score
hp_scores = [(c["candidate_id"], c["profile"]["current_title"], honeypot_score(c))
             for c in candidates]
hp_scores.sort(key=lambda x: -x[2])
print("\nHoneypot scores (top 10):")
for cid, title, score in hp_scores[:10]:
    print(f"  {cid} {title:<35} hp={score:.3f}")
