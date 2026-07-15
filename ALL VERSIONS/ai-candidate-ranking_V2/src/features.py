import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def semantic_similarity(jd_emb, cand_emb):
    return cosine_similarity([jd_emb], [cand_emb])[0][0]

def skill_match(jd_text, skills):
    jd_words = set(jd_text.lower().split())
    skills_set = set([s.lower() for s in skills.split(",")])
    return len(jd_words & skills_set) / (len(skills_set) + 1e-5)

def experience_score(exp_years):
    return min(exp_years / 10, 1.0)