import pandas as pd
from embedding import get_embedding
from features import semantic_similarity, skill_match, experience_score
from ranker import compute_score

# Load data
jd_text = open("../data/jd.txt").read()
df = pd.read_csv("../data/candidates.csv")

jd_emb = get_embedding(jd_text)

scores = []

for _, row in df.iterrows():
    cand_text = row["resume_text"]
    cand_emb = get_embedding(cand_text)

    sim = semantic_similarity(jd_emb, cand_emb)
    skill = skill_match(jd_text, row["skills"])
    exp = experience_score(row["experience_years"])

    final_score = compute_score(sim, skill, exp)

    scores.append(final_score)

df["score"] = scores
df = df.sort_values(by="score", ascending=False)
df["rank"] = range(1, len(df) + 1)

df.to_csv("../outputs/ranked_candidates.csv", index=False)

print("Ranking completed. Check outputs/")