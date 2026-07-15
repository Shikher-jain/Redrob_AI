"""
reasoning.py — Generate specific, honest reasoning strings for each ranked candidate.

The spec says reasoning is evaluated on:
1. Specific facts from the profile (not generic praise)
2. JD connection
3. Honest concerns where they exist
4. No hallucination
5. Variation across candidates
6. Rank-consistent tone

We build reasoning from actual feature values and raw candidate data.
"""

from __future__ import annotations
from config import PURE_CONSULTING, PREFERRED_LOCATIONS
from utils import clamp, norm, parse_date, days_since


def _yoe_phrase(yoe: float) -> str:
    if yoe < 3:
        return f"only {yoe:.1f} years of experience (below JD's 5-year minimum)"
    elif yoe < 5:
        return f"{yoe:.1f} years — below the preferred 5–9 year band"
    elif yoe <= 9:
        return f"{yoe:.1f} years in the JD's preferred 5–9 year band"
    else:
        return f"{yoe:.1f} years (above preferred band, may be overqualified)"


def _location_phrase(profile: dict, sig: dict) -> str:
    loc     = profile.get("location", "unknown")
    country = norm(profile.get("country", ""))
    notice  = int(sig.get("notice_period_days", 90))
    relocate = bool(sig.get("willing_to_relocate", False))

    loc_n = norm(loc)
    if any(p in loc_n for p in PREFERRED_LOCATIONS):
        loc_part = f"based in {loc} (preferred city)"
    elif country in ("india", "in"):
        rel_str = "willing to relocate" if relocate else "not willing to relocate"
        loc_part = f"in India ({loc}), {rel_str}"
    else:
        loc_part = f"outside India ({loc}) — visa sponsorship unavailable"

    notice_part = (
        "immediately available" if notice == 0
        else f"{notice}-day notice period"
    )
    return f"{loc_part}; {notice_part}"


def _top_skills(c: dict, n: int = 3) -> str:
    """Return top N credible skills (advanced/expert with evidence)."""
    skilled = [
        s for s in c.get("skills", [])
        if s.get("proficiency") in ("advanced", "expert")
        and (s.get("endorsements", 0) > 0 or s.get("duration_months", 0) > 6)
    ]
    skilled.sort(key=lambda s: (
        s.get("endorsements", 0) + s.get("duration_months", 0)
    ), reverse=True)
    names = [s["name"] for s in skilled[:n]]
    return ", ".join(names) if names else "no verified core skills"


def _career_summary(c: dict) -> str:
    career = c.get("career_history", [])
    if not career:
        return "no career history"
    current = career[0]
    company = current.get("company", "unknown")
    title   = current.get("title", "unknown")
    dur     = current.get("duration_months", 0)
    consulting = any(f in norm(company) for f in PURE_CONSULTING)
    co_tag = " (consulting firm)" if consulting else ""
    return f"currently {title} at {company}{co_tag} for {dur} months"


def _engagement_phrase(sig: dict) -> str:
    rr    = float(sig.get("recruiter_response_rate", 0.5))
    d     = days_since(parse_date(sig.get("last_active_date")))
    saved = int(sig.get("saved_by_recruiters_30d", 0))

    if d <= 7 and rr >= 0.7:
        return f"active in last {d}d, {rr:.0%} response rate, saved by {saved} recruiter(s)"
    elif d > 180:
        return f"inactive for {d} days — availability uncertain"
    elif rr < 0.2:
        return f"low recruiter response rate ({rr:.0%}) — hireability risk"
    else:
        return f"last active {d}d ago, {rr:.0%} response rate"


def generate_reasoning(rec: dict) -> str:
    """
    Generate a 1–2 sentence reasoning string grounded entirely in
    facts from the candidate's profile and features.
    No hallucination — every claim maps to a real field.
    """
    c       = rec.get("_raw_candidate", {})
    feats   = rec.get("features", {})
    rank    = rec.get("rank", 99)
    score   = rec.get("final_score", 0.0)

    if not c:
        return "Insufficient profile data for detailed reasoning."

    profile = c.get("profile", {})
    sig     = c.get("redrob_signals", {}) or {}

    yoe     = float(profile.get("years_of_experience", 0))
    title   = profile.get("current_title", "unknown title")
    sem     = feats.get("semantic_similarity", 0.0)
    cq      = feats.get("career_quality_score", 0.0)
    sm      = feats.get("skill_match_score", 0.0)
    be      = feats.get("behavioral_engagement_score", 0.0)
    hp      = feats.get("honeypot_score", 0.0)

    yoe_phrase  = _yoe_phrase(yoe)
    loc_phrase  = _location_phrase(profile, sig)
    skills_str  = _top_skills(c)
    career_str  = _career_summary(c)
    engage_str  = _engagement_phrase(sig)

    # ── Tier A: Strong fit (rank 1–15) ────────────────────────────────────
    if rank <= 15:
        concerns = []
        notice = int(sig.get("notice_period_days", 90))
        if notice > 60:
            concerns.append(f"{notice}-day notice period")
        if be < 0.4:
            concerns.append("lower platform engagement")

        concern_str = f" Concern: {'; '.join(concerns)}." if concerns else ""
        return (
            f"{title} with {yoe_phrase}; {career_str}. "
            f"Strong semantic alignment ({sem:.2f}) with verified skills in {skills_str}; "
            f"{loc_phrase}.{concern_str}"
        )

    # ── Tier B: Solid fit (rank 16–40) ────────────────────────────────────
    if rank <= 40:
        gap = "skill depth" if sm < 0.35 else ("career trajectory" if cq < 0.40 else "engagement signals")
        return (
            f"{title}, {yoe_phrase}; {career_str}. "
            f"Good JD semantic match ({sem:.2f}) with credible skills in {skills_str}; "
            f"gap area: {gap}. {loc_phrase}."
        )

    # ── Tier C: Partial fit (rank 41–70) ──────────────────────────────────
    if rank <= 70:
        strongest = max(
            [("semantic", sem), ("career quality", cq), ("skill match", sm)],
            key=lambda x: x[1]
        )
        return (
            f"{title} with {yoe_phrase}. Strongest signal is {strongest[0]} ({strongest[1]:.2f}); "
            f"overall score ({score:.3f}) reflects partial JD alignment. "
            f"{loc_phrase}; {engage_str}."
        )

    # ── Tier D: Weak fit (rank 71–100) ────────────────────────────────────
    primary_issue = []
    if cq < 0.25:
        primary_issue.append("career profile misaligns with AI/ML engineering role")
    if sm < 0.20:
        primary_issue.append("limited verified JD-relevant skills")
    if be < 0.30:
        primary_issue.append("low platform engagement")

    issue_str = "; ".join(primary_issue) if primary_issue else "limited overall fit"
    return (
        f"{title}, {yoe_phrase}. Included at rank {rank}: {issue_str}. "
        f"Semantic score {sem:.2f}; {engage_str}."
    )
