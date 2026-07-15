"""
feature_engineering.py
=======================
Derives interpretable numeric features from raw candidate dicts and the JD.

Design philosophy
-----------------
*  No keyword bag-of-words matching.  Every feature is either a
   structural signal (years of experience, company type) or derived from
   semantic text understanding done *outside* this module (embeddings).
*  Honeypot resistance: we sanity-check profiles for internally
   impossible timestamps/durations before scoring.
*  All feature functions return values in [0.0, 1.0] so the weighted
   combiner in ranker.py works on a consistent scale.
"""

from __future__ import annotations

import logging
import math
import re
from datetime import date, datetime
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# JD-derived constants (specific to the Senior AI Engineer role)
# ---------------------------------------------------------------------------

# Years of experience target band from the JD
JD_YOE_MIN = 5.0
JD_YOE_TARGET = 7.0  # sweet-spot stated in JD
JD_YOE_MAX = 9.0

# Salary band in INR LPA (mid-senior AI engineer in India)
SALARY_BAND_MIN = 20.0
SALARY_BAND_MAX = 60.0

# Notice period hard cutoff the JD calls out as preferred
NOTICE_PREFERRED_DAYS = 30

# Companies the JD explicitly disqualifies (pure consulting, no product work)
PURE_CONSULTING_FIRMS = {
    "tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini",
    "hcl", "tech mahindra", "mphasis", "hexaware",
}

# Skill names that map to the JD's "absolutely need" category
# (used as bonus signal; not used as a keyword gate)
CORE_SKILL_NAMES: set[str] = {
    "sentence-transformers", "sentence transformers",
    "embeddings", "vector search", "vector database",
    "faiss", "qdrant", "pinecone", "weaviate", "milvus", "opensearch",
    "elasticsearch",
    "retrieval", "rag", "retrieval augmented generation",
    "ranking", "learning to rank", "ltr",
    "information retrieval", "ir",
    "ndcg", "mrr", "a/b testing",
    "transformers", "huggingface", "bert", "roberta",
    "llm", "fine-tuning", "lora", "qlora",
    "python", "pytorch", "tensorflow",
    "hybrid search",
    "xgboost", "lightgbm",
}

PREFERRED_SKILL_NAMES: set[str] = {
    "distributed systems", "kafka", "spark",
    "open source", "nlp", "natural language processing",
}

# Industries / titles that clearly belong to IR/ML/AI engineering
ML_AI_INDUSTRIES = {
    "artificial intelligence", "machine learning", "nlp",
    "information retrieval", "data science", "ai", "ml",
    "tech", "technology", "software", "saas",
    "hr tech", "hrtech", "recruiting tech",
}

# Red-flag titles from the JD's "do NOT want" section
DISQUALIFIED_TITLE_TOKENS = {
    "marketing manager", "marketing",
    "operations manager",
    "hr manager", "human resources",
    "customer support", "support engineer",
    "graphic designer",
    "content writer", "seo",
    "business analyst",       # borderline — we penalise rather than eliminate
    "mechanical engineer",
    "chemical engineer",
}

# Tokens that strongly suggest research-only (JD disqualifier)
RESEARCH_ONLY_TOKENS = {"research scientist", "research engineer", "phd researcher"}

# Preferred locations (Pune / Noida / NCR / Hyderabad / Mumbai / Bangalore)
PREFERRED_LOCATIONS = {
    "pune", "noida", "delhi", "ncr", "gurgaon", "gurugram",
    "hyderabad", "mumbai", "bombay", "bangalore", "bengaluru",
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _today() -> date:
    return datetime.utcnow().date()


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def _normalise_text(text: str) -> str:
    return text.lower().strip()


def _company_is_pure_consulting(company: str) -> bool:
    name = _normalise_text(company)
    return any(firm in name for firm in PURE_CONSULTING_FIRMS)


def _title_is_disqualified(title: str) -> bool:
    t = _normalise_text(title)
    return any(tok in t for tok in DISQUALIFIED_TITLE_TOKENS)


def _is_research_only(title: str) -> bool:
    t = _normalise_text(title)
    return any(tok in t for tok in RESEARCH_ONLY_TOKENS)


def _date_from_str(s: str | None) -> date | None:
    if not s:
        return None
    try:
        return datetime.strptime(s[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Honeypot detection
# ---------------------------------------------------------------------------

def honeypot_probability(candidate: dict) -> float:
    """
    Return a probability in [0, 1] that this profile is a honeypot.

    Honeypots have subtly impossible internal contradictions:
    - years_of_experience > company_age_implied_by_dates
    - skill marked "expert" with 0 months used and 0 endorsements
    - duration_months inconsistent with start/end dates by large margins
    - total claimed experience vastly exceeds stated years_of_experience
    """
    signals = []
    today = _today()

    career: list[dict] = candidate.get("career_history", [])

    # 1. Check date consistency in career history
    for role in career:
        start = _date_from_str(role.get("start_date"))
        end_s = role.get("end_date")
        end = _date_from_str(end_s) if end_s else today
        stated_months = role.get("duration_months", 0)

        if start and end and start <= end:
            actual_months = (end.year - start.year) * 12 + (end.month - start.month)
            discrepancy = abs(actual_months - stated_months)
            # >24 months discrepancy is suspicious
            if discrepancy > 24:
                signals.append(min(1.0, discrepancy / 60.0))

        # Future start date
        if start and start > today:
            signals.append(0.8)

    # 2. "Expert" skill with 0 months AND 0 endorsements
    for skill in candidate.get("skills", []):
        if (
            skill.get("proficiency") == "expert"
            and skill.get("duration_months", 0) == 0
            and skill.get("endorsements", 0) == 0
        ):
            signals.append(0.6)

    # 3. Excessive claimed skills in "expert" tier (>10 expert skills total)
    expert_count = sum(
        1 for s in candidate.get("skills", [])
        if s.get("proficiency") == "expert"
    )
    if expert_count > 10:
        signals.append(min(0.9, (expert_count - 10) * 0.08))

    if not signals:
        return 0.0
    # Combine: if any single signal is strong, overall is penalised
    return _clamp(max(signals) * 0.7 + (sum(signals) / len(signals)) * 0.3)


# ---------------------------------------------------------------------------
# Feature 1: Experience Score
# ---------------------------------------------------------------------------

def experience_score(candidate: dict) -> float:
    """
    Score how well the candidate's experience profile fits the JD.

    Sub-components:
      - years of experience (sweet-spot penalty outside 5-9 band)
      - fraction of career spent at product companies vs consulting
      - current title alignment
      - evidence of production deployment (inferred from role descriptions)
      - disqualifier penalties (pure consulting only, research-only, etc.)
    """
    profile = candidate.get("profile", {})
    career: list[dict] = candidate.get("career_history", [])

    yoe = float(profile.get("years_of_experience", 0))
    current_title = profile.get("current_title", "")

    # -- Years of experience component --
    if yoe < 3:
        yoe_score = 0.1
    elif yoe < JD_YOE_MIN:
        yoe_score = 0.5 + 0.3 * (yoe - 3) / (JD_YOE_MIN - 3)
    elif yoe <= JD_YOE_MAX:
        # Peak at JD_YOE_TARGET, gentle falloff either side
        dist = abs(yoe - JD_YOE_TARGET) / (JD_YOE_MAX - JD_YOE_MIN)
        yoe_score = 1.0 - 0.25 * dist
    else:
        # Beyond max — not disqualified, just less ideal
        yoe_score = max(0.5, 1.0 - 0.05 * (yoe - JD_YOE_MAX))

    # -- Product vs consulting career component --
    total_months = 0
    consulting_months = 0
    product_company_months = 0

    for role in career:
        dur = role.get("duration_months", 0)
        total_months += dur
        company = role.get("company", "")
        if _company_is_pure_consulting(company):
            consulting_months += dur
        else:
            product_company_months += dur

    if total_months > 0:
        consulting_fraction = consulting_months / total_months
        # JD says "exclusively consulting" is a disqualifier; any product experience is fine
        product_score = 1.0 - 0.7 * consulting_fraction
    else:
        product_score = 0.5

    # -- Title alignment component --
    title_score = 1.0
    if _is_research_only(current_title):
        title_score = 0.2
    elif _title_is_disqualified(current_title):
        title_score = 0.1

    # -- Evidence of production deployment in descriptions --
    production_keywords = {
        "production", "deployed", "shipped", "live", "users", "scale",
        "latency", "throughput", "api", "service", "pipeline", "real-time",
        "real time", "endpoint", "inference",
    }
    desc_text = " ".join(
        _normalise_text(r.get("description", "")) for r in career
    )
    production_hits = sum(1 for kw in production_keywords if kw in desc_text)
    production_score = _clamp(production_hits / 6.0)

    # Weighted combination
    score = (
        0.35 * yoe_score
        + 0.25 * product_score
        + 0.20 * title_score
        + 0.20 * production_score
    )
    return _clamp(score)


# ---------------------------------------------------------------------------
# Feature 2: Skill Match Score
# ---------------------------------------------------------------------------

def skill_match_score(candidate: dict) -> float:
    """
    Evaluate skill fit using endorsements + duration, not just presence.

    A skill listed with 'expert' proficiency, 36 months duration, and 15
    endorsements counts for much more than 'beginner' with 0 months.

    Honeypot resistance: we apply a trust multiplier based on duration × endorsements.
    """
    skills: list[dict] = candidate.get("skills", [])
    assessments: dict[str, float] = (
        candidate.get("redrob_signals", {}).get("skill_assessment_scores", {}) or {}
    )

    if not skills:
        return 0.0

    PROFICIENCY_WEIGHT = {"beginner": 0.25, "intermediate": 0.5, "advanced": 0.8, "expert": 1.0}
    CORE_BONUS = 1.6
    PREFERRED_BONUS = 1.2

    weighted_core = 0.0
    weighted_preferred = 0.0
    max_possible_core = 0.0
    max_possible_pref = 0.0

    for skill in skills:
        name_raw = skill.get("name", "")
        name = _normalise_text(name_raw)
        proficiency = PROFICIENCY_WEIGHT.get(skill.get("proficiency", "beginner"), 0.25)
        endorsements = skill.get("endorsements", 0)
        duration = skill.get("duration_months", 0)

        # Trust multiplier: endorsements and months-used validate the claim
        # Caps at 1.0 so it's a credibility check, not an amplifier
        trust = _clamp(
            0.5 * min(1.0, endorsements / 10.0)
            + 0.5 * min(1.0, duration / 24.0)
        )

        # Override with assessment score if available (objective evidence)
        assessment_val = assessments.get(name_raw, assessments.get(name, -1))
        if assessment_val >= 0:
            # Assessment replaces self-reported proficiency
            effective_proficiency = _clamp(assessment_val / 100.0)
            trust = 1.0  # Assessed = fully trusted
        else:
            effective_proficiency = proficiency

        skill_value = effective_proficiency * trust

        is_core = any(core in name for core in CORE_SKILL_NAMES)
        is_preferred = any(pref in name for pref in PREFERRED_SKILL_NAMES)

        if is_core:
            weighted_core += skill_value * CORE_BONUS
            max_possible_core += CORE_BONUS
        elif is_preferred:
            weighted_preferred += skill_value * PREFERRED_BONUS
            max_possible_pref += PREFERRED_BONUS

    # Normalise core and preferred components separately
    core_norm = weighted_core / max(max_possible_core, 1.0)
    pref_norm = weighted_preferred / max(max_possible_pref, 1.0)

    # Overall: core skills dominate, preferred provide a smaller boost
    score = 0.75 * _clamp(core_norm) + 0.25 * _clamp(pref_norm)
    return _clamp(score)


# ---------------------------------------------------------------------------
# Feature 3: Location & Availability Score
# ---------------------------------------------------------------------------

def location_availability_score(candidate: dict) -> float:
    """
    Score based on:
    - Location match (Pune/Noida/NCR preferred; India required; overseas penalised)
    - Willing to relocate
    - Notice period (< 30 days preferred; > 90 days penalised)
    - open_to_work_flag
    - last_active_date recency
    """
    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})

    location = _normalise_text(profile.get("location", ""))
    country = _normalise_text(profile.get("country", ""))
    willing_to_relocate = signals.get("willing_to_relocate", False)
    open_to_work = signals.get("open_to_work_flag", False)
    notice_days = signals.get("notice_period_days", 90)

    # -- Location component --
    if any(loc in location for loc in PREFERRED_LOCATIONS):
        loc_score = 1.0
    elif country in ("india", "in"):
        # In India but not preferred city — relocate bump
        loc_score = 0.6 + 0.3 * float(willing_to_relocate)
    elif country in ("", "unknown"):
        loc_score = 0.4
    else:
        # Outside India — JD says "case-by-case, no visa sponsorship"
        loc_score = 0.2 + 0.2 * float(willing_to_relocate)

    # -- Notice period component --
    if notice_days <= 0:
        notice_score = 1.0  # Immediately available
    elif notice_days <= NOTICE_PREFERRED_DAYS:
        notice_score = 0.9
    elif notice_days <= 60:
        notice_score = 0.6
    elif notice_days <= 90:
        notice_score = 0.4
    else:
        notice_score = 0.2

    # -- Availability component --
    avail_score = 0.5
    if open_to_work:
        avail_score += 0.3

    last_active = _date_from_str(signals.get("last_active_date"))
    if last_active:
        days_inactive = (datetime.utcnow().date() - last_active).days
        if days_inactive <= 7:
            avail_score += 0.2
        elif days_inactive <= 30:
            avail_score += 0.1
        elif days_inactive > 180:
            avail_score -= 0.2  # Cold profile — JD explicitly calls this out

    avail_score = _clamp(avail_score)

    return _clamp(0.40 * loc_score + 0.30 * notice_score + 0.30 * avail_score)


# ---------------------------------------------------------------------------
# Feature 4: Behavioral Engagement Score
# ---------------------------------------------------------------------------

def behavioral_engagement_score(candidate: dict) -> float:
    """
    Score derived from the 23 Redrob behavioral signals.

    The JD explicitly states: "a perfect-on-paper candidate who hasn't
    logged in for 6 months and has a 5% recruiter response rate is, for
    hiring purposes, not actually available."

    Components:
    - Recruiter response rate
    - Profile completeness
    - GitHub activity
    - Saved by recruiters (market validation signal)
    - Interview completion rate
    - Verification signals (email, phone, LinkedIn)
    """
    s = candidate.get("redrob_signals", {})
    if not s:
        return 0.5  # No signals — neutral

    response_rate = float(s.get("recruiter_response_rate", 0.5))
    profile_complete = float(s.get("profile_completeness_score", 50)) / 100.0
    github_raw = float(s.get("github_activity_score", -1))
    saved_30d = int(s.get("saved_by_recruiters_30d", 0))
    interview_rate = float(s.get("interview_completion_rate", 0.5))
    verified_email = float(s.get("verified_email", False))
    verified_phone = float(s.get("verified_phone", False))
    linkedin = float(s.get("linkedin_connected", False))
    avg_response_h = float(s.get("avg_response_time_hours", 48))

    # Github: -1 means not linked; treat as 0
    github_score = _clamp(github_raw / 100.0) if github_raw >= 0 else 0.0

    # Saved by recruiters: market validation — normalise on 0–10 scale
    saved_score = _clamp(saved_30d / 10.0)

    # Response time: < 4h is great; > 72h is poor
    if avg_response_h <= 4:
        response_time_score = 1.0
    elif avg_response_h <= 24:
        response_time_score = 0.7
    elif avg_response_h <= 72:
        response_time_score = 0.4
    else:
        response_time_score = 0.2

    # Verification bundle
    verification_score = (verified_email + verified_phone + linkedin) / 3.0

    score = (
        0.25 * response_rate
        + 0.15 * profile_complete
        + 0.15 * github_score
        + 0.15 * saved_score
        + 0.10 * interview_rate
        + 0.10 * response_time_score
        + 0.10 * verification_score
    )
    return _clamp(score)


# ---------------------------------------------------------------------------
# Feature 5: Education Score
# ---------------------------------------------------------------------------

def education_score(candidate: dict) -> float:
    """
    Light signal — less weight than career signals per the JD's philosophy.

    Components:
    - Institution tier (tier_1 > tier_2 > tier_3 > tier_4)
    - Field relevance (CS/IT/Math/Stats/EE preferred)
    - Degree level (M.Tech / M.S. slight bonus)
    """
    education: list[dict] = candidate.get("education", [])
    if not education:
        return 0.5

    TIER_SCORE = {"tier_1": 1.0, "tier_2": 0.75, "tier_3": 0.55, "tier_4": 0.35, "unknown": 0.45}
    RELEVANT_FIELDS = {
        "computer science", "cs", "information technology", "it",
        "mathematics", "statistics", "electrical engineering", "ee",
        "electronics", "data science", "artificial intelligence",
        "machine learning", "natural language processing", "nlp",
    }
    DEGREE_BONUS = {"m.tech": 0.1, "m.e.": 0.1, "m.s.": 0.1, "ms": 0.1,
                    "ph.d": 0.05, "phd": 0.05}  # PhD slight bonus (not research-only penalty)

    best_score = 0.0
    for edu in education:
        tier = edu.get("tier", "unknown")
        field = _normalise_text(edu.get("field_of_study", ""))
        degree = _normalise_text(edu.get("degree", ""))

        tier_val = TIER_SCORE.get(tier, 0.45)
        field_match = any(f in field for f in RELEVANT_FIELDS)
        field_val = 1.0 if field_match else 0.4

        degree_bonus = next((v for k, v in DEGREE_BONUS.items() if k in degree), 0.0)

        edu_score = _clamp(0.55 * tier_val + 0.45 * field_val + degree_bonus)
        best_score = max(best_score, edu_score)

    return _clamp(best_score)


# ---------------------------------------------------------------------------
# Feature 6: Salary Fit Score
# ---------------------------------------------------------------------------

def salary_fit_score(candidate: dict) -> float:
    """
    Check whether the candidate's expected salary is within the market band
    for a Senior AI Engineer in India.  Out-of-band candidates are not
    disqualified but get a slight penalty since negotiation risk is real.
    """
    signals = candidate.get("redrob_signals", {})
    sal = signals.get("expected_salary_range_inr_lpa", {})
    if not sal:
        return 0.7  # Unknown = neutral-ish

    sal_min = float(sal.get("min", 0))
    sal_max = float(sal.get("max", 0))
    sal_mid = (sal_min + sal_max) / 2.0

    if sal_mid == 0:
        return 0.7

    if SALARY_BAND_MIN <= sal_mid <= SALARY_BAND_MAX:
        return 1.0
    elif sal_mid < SALARY_BAND_MIN:
        # Below band — not a bad sign necessarily
        return 0.8
    else:
        # Above band — negotiation risk
        overshoot = (sal_mid - SALARY_BAND_MAX) / SALARY_BAND_MAX
        return _clamp(1.0 - overshoot)


# ---------------------------------------------------------------------------
# Master feature extractor
# ---------------------------------------------------------------------------

def extract_features(candidate: dict) -> dict[str, float]:
    """
    Return a dict of all engineered features for a single candidate.

    This is the only function called by ranker.py; all sub-features are
    encapsulated here.
    """
    honeypot_prob = honeypot_probability(candidate)

    return {
        # Core features (weighted in ranker.py)
        "experience_score": experience_score(candidate),
        "skill_match_score": skill_match_score(candidate),
        "location_availability_score": location_availability_score(candidate),
        "behavioral_engagement_score": behavioral_engagement_score(candidate),
        "education_score": education_score(candidate),
        "salary_fit_score": salary_fit_score(candidate),
        # Penalty/flag features
        "honeypot_probability": honeypot_prob,
        # Semantic score placeholder (filled in by ranker.py after embedding)
        "semantic_similarity": 0.0,
    }