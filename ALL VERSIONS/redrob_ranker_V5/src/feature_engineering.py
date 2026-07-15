"""
feature_engineering.py — Structured features from candidate profile + behavioral signals.

All features return float in [0.0, 1.0].
No keyword bag-of-words as primary logic — skills are scored by
proficiency × credibility (endorsements + duration).
"""

from __future__ import annotations
import logging

from config import (
    CORE_SKILLS, DISQUALIFIED_TITLE_TOKENS, JD_YOE_MAX, JD_YOE_MIN,
    JD_YOE_TARGET, ML_INDUSTRIES, NOTICE_PREFERRED_DAYS, PREFERRED_LOCATIONS,
    PREFERRED_SKILLS, PRODUCTION_KEYWORDS, PURE_CONSULTING,
    RESEARCH_ONLY_TOKENS, SALARY_MAX_LPA, SALARY_MIN_LPA, WRONG_DOMAIN_TOKENS,
)
from utils import clamp, contains_any, days_since, norm, parse_date, today

logger = logging.getLogger(__name__)


# ── Honeypot Detection ────────────────────────────────────────────────────────

def honeypot_score(c: dict) -> float:
    """
    Return [0,1] probability this is a honeypot profile.
    Honeypots have internally impossible contradictions.
    """
    flags: list[float] = []
    td = today()

    for role in c.get("career_history", []):
        start = parse_date(role.get("start_date"))
        end   = parse_date(role.get("end_date")) if role.get("end_date") else td

        if start and end and start <= end:
            actual_m = (end.year - start.year) * 12 + (end.month - start.month)
            stated_m = role.get("duration_months", 0)
            disc = abs(actual_m - stated_m)
            if disc > 24:
                flags.append(min(1.0, disc / 60.0))

        if start and start > td:
            flags.append(0.9)  # future start date

    expert_count = 0
    for skill in c.get("skills", []):
        if skill.get("proficiency") == "expert":
            expert_count += 1
            if skill.get("duration_months", 0) == 0 and skill.get("endorsements", 0) == 0:
                flags.append(0.7)   # expert with zero evidence = red flag

    if expert_count > 12:
        flags.append(min(0.9, (expert_count - 12) * 0.07))

    if not flags:
        return 0.0
    return clamp(max(flags) * 0.6 + (sum(flags) / len(flags)) * 0.4)


# ── Feature 1: Career Quality ─────────────────────────────────────────────────

def career_quality_score(c: dict) -> float:
    """
    Evaluates:
    a) Years of experience vs JD sweet-spot (5–9 years)
    b) Product company fraction (consulting = penalty)
    c) Current title alignment (disqualified titles = near-zero)
    d) Production deployment evidence in role descriptions
    e) Job-hopping penalty
    """
    profile = c.get("profile", {})
    career  = c.get("career_history", [])
    yoe     = float(profile.get("years_of_experience", 0))
    title   = profile.get("current_title", "")

    # a) YOE
    if yoe < 2:
        yoe_s = 0.05
    elif yoe < JD_YOE_MIN:
        yoe_s = 0.30 + 0.40 * (yoe - 2) / (JD_YOE_MIN - 2)
    elif yoe <= JD_YOE_MAX:
        dist  = abs(yoe - JD_YOE_TARGET)
        yoe_s = 1.0 - 0.20 * (dist / (JD_YOE_MAX - JD_YOE_MIN))
    else:
        yoe_s = max(0.55, 1.0 - 0.04 * (yoe - JD_YOE_MAX))

    # b) Product vs consulting
    total_m = sum(r.get("duration_months", 0) for r in career)
    consult_m = sum(
        r.get("duration_months", 0) for r in career
        if any(f in norm(r.get("company", "")) for f in PURE_CONSULTING)
    )
    if total_m > 0:
        cf = consult_m / total_m
        product_s = 0.05 if cf >= 0.95 else (1.0 - 0.7 * cf)
    else:
        product_s = 0.5

    # c) Title
    if contains_any(title, RESEARCH_ONLY_TOKENS):
        title_s = 0.15
    elif contains_any(title, DISQUALIFIED_TITLE_TOKENS):
        title_s = 0.05
    elif contains_any(title, WRONG_DOMAIN_TOKENS):
        title_s = 0.25
    else:
        title_s = 1.0

    # d) Production evidence in descriptions
    desc_blob = " ".join(norm(r.get("description", "")) for r in career)
    hits  = sum(1 for kw in PRODUCTION_KEYWORDS if kw in desc_blob)
    prod_s = clamp(hits / 7.0)

    # e) Job-hopping
    short = sum(1 for r in career if r.get("duration_months", 99) < 12)
    hop_penalty = clamp(short / max(len(career), 1) * 0.4)

    score = (
        0.30 * yoe_s
        + 0.25 * product_s
        + 0.25 * title_s
        + 0.20 * prod_s
        - 0.15 * hop_penalty
    )
    return clamp(score)


# ── Feature 2: Skill Match ────────────────────────────────────────────────────

def skill_match_score(c: dict) -> float:
    """
    Scores skills by proficiency × credibility (endorsements + duration).

    Anti-keyword-stuffer: expert skill with 0 months and 0 endorsements
    gets near-zero weight regardless of the label.

    Platform assessment scores override self-reported proficiency.
    """
    skills      = c.get("skills", [])
    assessments = c.get("redrob_signals", {}).get("skill_assessment_scores") or {}
    if not skills:
        return 0.0

    PROF_W = {"beginner": 0.20, "intermediate": 0.45, "advanced": 0.75, "expert": 1.0}

    core_w, core_max = 0.0, 0.0
    pref_w, pref_max = 0.0, 0.0

    for skill in skills:
        name_raw  = skill.get("name", "")
        name_n    = norm(name_raw)
        prof      = PROF_W.get(skill.get("proficiency", "beginner"), 0.20)
        end_count = skill.get("endorsements", 0)
        dur       = skill.get("duration_months", 0)

        # Credibility: capped at 1.0 (validation, not amplifier)
        trust = clamp(0.5 * min(1.0, end_count / 15.0) + 0.5 * min(1.0, dur / 30.0))

        # Assessment overrides self-reported
        akey = next((k for k in assessments if norm(k) == name_n), None)
        if akey:
            prof  = clamp(assessments[akey] / 100.0)
            trust = 1.0   # objective test = fully trusted

        effective = prof * trust

        is_core = any(cs in name_n for cs in CORE_SKILLS)
        is_pref = any(ps in name_n for ps in PREFERRED_SKILLS)

        if is_core:
            core_w   += effective * 2.0
            core_max += 2.0
        elif is_pref:
            pref_w   += effective
            pref_max += 1.0

    cn = core_w / max(core_max, 1.0)
    pn = pref_w / max(pref_max, 1.0)
    return clamp(0.80 * cn + 0.20 * pn)


# ── Feature 3: Location & Availability ───────────────────────────────────────

def location_availability_score(c: dict) -> float:
    """
    - Preferred India cities score 1.0
    - India non-preferred + willing to relocate = 0.90
    - Outside India (no visa sponsorship per JD) = 0.15–0.30
    - Notice period: sub-30 = 0.90, 30-60 = 0.55, 60-90 = 0.35, >90 = 0.15
    - open_to_work + recent activity boost availability component
    """
    profile  = c.get("profile", {})
    sig      = c.get("redrob_signals", {})

    location = norm(profile.get("location", ""))
    country  = norm(profile.get("country", ""))
    relocate = bool(sig.get("willing_to_relocate", False))
    otw      = bool(sig.get("open_to_work_flag", False))
    notice   = int(sig.get("notice_period_days", 90))

    # Location
    if any(loc in location for loc in PREFERRED_LOCATIONS):
        loc_s = 1.0
    elif country in ("india", "in"):
        loc_s = 0.65 + 0.25 * float(relocate)
    else:
        loc_s = 0.15 + 0.15 * float(relocate)

    # Notice
    if notice == 0:
        notice_s = 1.0
    elif notice <= NOTICE_PREFERRED_DAYS:
        notice_s = 0.90
    elif notice <= 60:
        notice_s = 0.55
    elif notice <= 90:
        notice_s = 0.35
    else:
        notice_s = 0.15

    # Platform availability
    avail_s = 0.40 + 0.30 * float(otw)
    d = days_since(parse_date(sig.get("last_active_date")))
    if d <= 7:
        avail_s += 0.30
    elif d <= 30:
        avail_s += 0.20
    elif d <= 90:
        avail_s += 0.10
    elif d > 180:
        avail_s -= 0.20

    return clamp(0.40 * loc_s + 0.30 * notice_s + 0.30 * clamp(avail_s))


# ── Feature 4: Behavioral Engagement ─────────────────────────────────────────

def behavioral_engagement_score(c: dict) -> float:
    """
    From the JD: "a perfect-on-paper candidate who hasn't logged in for 6 months
    and has a 5% recruiter response rate is not actually available."

    Components (with weights):
    - recruiter_response_rate  0.25 — most predictive of reachability
    - saved_by_recruiters_30d  0.15 — market validation
    - github_activity_score    0.15 — external coding signal
    - profile_completeness     0.10
    - interview_completion_rate 0.10
    - avg_response_time_hours  0.10
    - verification bundle      0.10
    - profile_views_30d        0.05
    """
    s = c.get("redrob_signals", {})
    if not s:
        return 0.5

    resp_rate  = clamp(float(s.get("recruiter_response_rate", 0.5)))
    complete   = clamp(float(s.get("profile_completeness_score", 50)) / 100.0)
    gh_raw     = float(s.get("github_activity_score", -1))
    saved_30d  = int(s.get("saved_by_recruiters_30d", 0))
    int_rate   = clamp(float(s.get("interview_completion_rate", 0.5)))
    v_email    = float(bool(s.get("verified_email", False)))
    v_phone    = float(bool(s.get("verified_phone", False)))
    linkedin   = float(bool(s.get("linkedin_connected", False)))
    resp_h     = float(s.get("avg_response_time_hours", 72))
    views_30d  = int(s.get("profile_views_received_30d", 0))

    gh_s    = clamp(gh_raw / 100.0) if gh_raw >= 0 else 0.0
    saved_s = clamp(saved_30d / 8.0)

    if resp_h <= 4:
        resp_t_s = 1.0
    elif resp_h <= 24:
        resp_t_s = 0.75
    elif resp_h <= 72:
        resp_t_s = 0.45
    else:
        resp_t_s = 0.15

    verify_s = (v_email + v_phone + linkedin) / 3.0
    views_s  = clamp(views_30d / 30.0)

    return clamp(
        0.25 * resp_rate
        + 0.15 * saved_s
        + 0.15 * gh_s
        + 0.10 * complete
        + 0.10 * int_rate
        + 0.10 * resp_t_s
        + 0.10 * verify_s
        + 0.05 * views_s
    )


# ── Master Extractor ──────────────────────────────────────────────────────────

def extract_all_features(c: dict) -> dict[str, float]:
    """
    Return all engineered features for one candidate.
    `semantic_similarity` is 0.0 here; ranker.py fills it after embedding.
    
    Career-skill coherence multiplier is applied to skill_match_score here:
    candidates who claim AI skills but have zero AI work in their descriptions
    get a penalty that reduces their effective skill score.
    """
    coherence    = _career_skill_coherence(c)
    raw_skill_ms = skill_match_score(c)

    return {
        "career_quality_score":        career_quality_score(c),
        "skill_match_score":           raw_skill_ms * coherence,   # penalise incoherent claims
        "location_availability_score": location_availability_score(c),
        "behavioral_engagement_score": behavioral_engagement_score(c),
        "honeypot_score":              honeypot_score(c),
        "career_skill_coherence":      coherence,
        "semantic_similarity":         0.0,
    }


# ── Career-Skill Coherence Check (added fix) ─────────────────────────────────

def _career_skill_coherence(c: dict) -> float:
    """
    Cross-validate claimed skills against career history descriptions.
    
    A "Frontend Engineer at Zomato" claiming FAISS expert (40 endorsements)
    should be penalised if their descriptions never mention ML/IR work.
    
    Returns a multiplier in [0.4, 1.0]:
    - 1.0  = skills and career descriptions are coherent
    - 0.4  = major mismatch (AI skills claimed, zero AI work in descriptions)
    """
    skills = c.get('skills', [])
    career = c.get('career_history', [])
    
    # Check if candidate claims significant AI/ML skills
    ai_skill_names = {norm(s['name']) for s in skills 
                      if s.get('proficiency') in ('advanced','expert')
                      and any(cs in norm(s['name']) for cs in CORE_SKILLS)}
    
    if not ai_skill_names:
        return 1.0  # No AI skills claimed → no incoherence possible
    
    # Check if descriptions back up AI work
    desc_blob = " ".join(norm(r.get('description','')) for r in career)
    ai_desc_hits = sum(1 for kw in {
        'machine learning', 'deep learning', 'neural', 'model', 'embedding',
        'retrieval', 'recommendation', 'ranking', 'nlp', 'natural language',
        'vector', 'classification', 'prediction', 'training', 'inference',
        'data science', 'ai ', 'ml ', 'algorithm', 'feature engineering',
    } if kw in desc_blob)
    
    # If strong AI skills claimed but career descriptions have no AI work: suspicious
    if len(ai_skill_names) >= 2 and ai_desc_hits < 2:
        return 0.45   # Significant mismatch
    elif len(ai_skill_names) >= 1 and ai_desc_hits < 1:
        return 0.60   # Mild mismatch
    
    return 1.0
