"""
Nova Candidate Map — v3 backend
Pipeline: Jina → O*NET → CareerOneStop → Tavily → Gemini/Groq → normalize
"""
import os, re, json, time, urllib.parse
import requests

# ── API keys (set as Vercel env vars) ──────────────────────────────────────
JINA_KEY        = os.environ.get("JINA_API_KEY", "")
ONET_USER       = os.environ.get("ONET_USERNAME", "")
ONET_PASS       = os.environ.get("ONET_PASSWORD", "")
CAREERONESTOP_USER  = os.environ.get("CAREERONESTOP_USER_ID", "")
CAREERONESTOP_TOKEN = os.environ.get("CAREERONESTOP_TOKEN", "")
TAVILY_KEY      = os.environ.get("TAVILY_API_KEY", "")
GEMINI_KEY      = os.environ.get("GEMINI_API_KEY", "")
GROQ_KEY        = os.environ.get("GROQ_API_KEY", "")

TIMEOUT = 20


# ── Stage 1: Jina content fetch ────────────────────────────────────────────

def _extract_title_from_url(url: str) -> str:
    """Fallback: pull job title from URL slug."""
    try:
        path = urllib.parse.urlparse(url).path
        slug = path.strip("/").split("/")[-1]
        title = re.sub(r"[-_]", " ", slug).strip()
        # Remove common noise words
        title = re.sub(r"\b(job|jobs|career|careers|apply|posting|opening)\b", "", title, flags=re.I).strip()
        return title.title() or "Unknown Role"
    except Exception:
        return "Unknown Role"


def _fetch_url(url: str) -> dict:
    """Fetch job posting via Jina Reader. Returns {content, fallback_triggered, inferred_title}."""
    try:
        headers = {"Accept": "text/plain"}
        if JINA_KEY:
            headers["Authorization"] = f"Bearer {JINA_KEY}"
        resp = requests.get(f"https://r.jina.ai/{url}", headers=headers, timeout=TIMEOUT)
        content = resp.text if resp.status_code == 200 else ""
    except Exception:
        content = ""

    if len(content) < 200:
        return {
            "content": content,
            "fallback_triggered": True,
            "inferred_title": _extract_title_from_url(url),
        }
    return {"content": content, "fallback_triggered": False, "inferred_title": ""}


# ── Stage 2: O*NET occupation grounding ────────────────────────────────────

_ONET_CACHE = {}

def _onet_grounding(title: str) -> dict:
    """Look up occupation via O*NET. Returns ONET_GROUNDING dict."""
    if not title or not (ONET_USER and ONET_PASS):
        return {}
    if title in _ONET_CACHE:
        return _ONET_CACHE[title]

    try:
        # Search for best match
        search = requests.get(
            "https://services.onetcenter.org/ws/mnm/search",
            params={"keyword": title, "end": 1},
            auth=(ONET_USER, ONET_PASS),
            headers={"Accept": "application/json"},
            timeout=TIMEOUT,
        )
        if search.status_code != 200:
            return {}
        data = search.json()
        occupations = data.get("occupation", [])
        if not occupations:
            return {}

        soc_code = occupations[0].get("code", "")
        title_match = occupations[0].get("title", title)

        # Fetch summary
        summary = requests.get(
            f"https://services.onetcenter.org/ws/online/occupations/{soc_code}",
            auth=(ONET_USER, ONET_PASS),
            headers={"Accept": "application/json"},
            timeout=TIMEOUT,
        )
        summary_data = summary.json() if summary.status_code == 200 else {}

        result = {
            "soc_code": soc_code,
            "matched_title": title_match,
            "education_distribution": summary_data.get("education", {}),
            "skills": [s.get("name") for s in summary_data.get("skills", {}).get("element", [])[:8]],
        }
        _ONET_CACHE[title] = result
        return result

    except Exception:
        return {}


# ── Stage 3: CareerOneStop wage grounding ──────────────────────────────────

def _careeronestop_wages(title: str, location: str) -> dict:
    """Fetch 25th/50th/75th wage percentiles. Returns MARKET_GROUNDING.salary_bounds."""
    if not title or not (CAREERONESTOP_USER and CAREERONESTOP_TOKEN):
        return {}

    try:
        keyword = urllib.parse.quote(title)
        loc = urllib.parse.quote(location or "National")
        url = (
            f"https://api.careeronestop.org/v1/comparesalaries/{CAREERONESTOP_USER}"
            f"/{keyword}/{loc}/0/5"
        )
        resp = requests.get(
            url,
            headers={
                "Authorization": f"Bearer {CAREERONESTOP_TOKEN}",
                "Accept": "application/json",
            },
            timeout=TIMEOUT,
        )
        if resp.status_code != 200:
            return {}

        data = resp.json()
        occ_list = data.get("OccupationList", [])
        if not occ_list:
            return {}

        wages = occ_list[0].get("Wages", {})
        annual = wages.get("NationalWagesList", [{}])[0] if wages else {}

        return {
            "percentile_25": annual.get("Pct25", ""),
            "median": annual.get("Median", ""),
            "percentile_75": annual.get("Pct75", ""),
            "location": location or "National",
        }

    except Exception:
        return {}


# ── Stage 4: Tavily demographic research ──────────────────────────────────

def _tavily_demographics(title: str, location: str) -> str:
    """Search for demographic/labor pool data. Returns compact string."""
    if not TAVILY_KEY or not title:
        return ""

    try:
        query = f'"{title}" candidate demographics labor pool characteristics'
        if location:
            query += f" {location}"

        resp = requests.post(
            "https://api.tavily.com/search",
            json={
                "api_key": TAVILY_KEY,
                "query": query,
                "max_results": 3,
                "search_depth": "basic",
            },
            timeout=TIMEOUT,
        )
        if resp.status_code != 200:
            return ""

        results = resp.json().get("results", [])
        snippets = [f"{r['title']}: {r['content'][:300]}" for r in results if r.get("content")]
        return "\n\n".join(snippets[:3])

    except Exception:
        return ""


# ── Stage 5: Signal extraction ─────────────────────────────────────────────

def _extract_signals(text: str) -> dict:
    """Extract title, company, location, salary, arrangement from JD text."""
    signals = {"title": "", "company": "", "location": "", "salary": "", "arrangement": ""}

    # Title: look for common patterns
    title_patterns = [
        r"(?:job title|position|role)[:\s]+([A-Za-z][^\n,|]{3,60})",
        r"^#+\s*([A-Za-z][^\n]{3,60})$",
        r"^([A-Za-z][^\n]{3,60})\n",
    ]
    for pat in title_patterns:
        m = re.search(pat, text[:500], re.MULTILINE | re.IGNORECASE)
        if m:
            signals["title"] = m.group(1).strip()
            break

    # Location
    loc_m = re.search(
        r"\b([A-Z][a-z]+(?:\s[A-Z][a-z]+)*,\s*(?:[A-Z]{2}|[A-Za-z]+))\b", text[:1000]
    )
    if loc_m:
        signals["location"] = loc_m.group(1)

    # Salary
    sal_m = re.search(
        r"[\$£€₹][\d,]+(?:\s*[-–]\s*[\$£€₹]?[\d,]+)?(?:\s*(?:per year|per hour|annually|\/yr|\/hr))?",
        text, re.IGNORECASE,
    )
    if sal_m:
        signals["salary"] = sal_m.group(0)

    # Arrangement
    if re.search(r"\b(?:remote|work from home|wfh)\b", text, re.IGNORECASE):
        signals["arrangement"] = "remote"
    elif re.search(r"\b(?:hybrid)\b", text, re.IGNORECASE):
        signals["arrangement"] = "hybrid"
    elif re.search(r"\b(?:on.?site|in.?office|in.?person)\b", text, re.IGNORECASE):
        signals["arrangement"] = "on-site"

    return signals


# ── Stage 5: LLM system prompt ─────────────────────────────────────────────

SYSTEM_PROMPT = """You are the core intelligence engine of Nova Candidate Map. Your purpose is to convert raw job description text and localized labor market data into an auditable, data-backed candidate market map consisting of 3 to 6 MECE personas.

STEP 0 — EXTRACT HARD JD FILTERS (do this before everything else)
Read the full JD and extract these constraints. Every persona you generate MUST be grounded in these filters:
1. work_location: exact city/country/remote status
2. shift_constraint: specific required hours or timezone (e.g. "US shift 5PM–2AM IST", "night shift", "UK hours")
3. onsite_requirement: mandatory on-site / hybrid / remote
4. experience_range: min–max years explicitly stated
5. preferred_industries: sectors/backgrounds explicitly mentioned or implied (e.g. "adtech", "SaaS CS", "recruitment marketing")
6. required_tools_metrics: specific tools, platforms, or metrics named (e.g. "CPA, CPC, CTR, CPH, NRR", "Salesforce", "programmatic")
7. explicit_disqualifiers: anything the JD says or implies will cause failure (e.g. "not for people who treat AM as status reporting", "on-site is non-negotiable", "night shift flexibility required")

CRITICAL RULES FOR PERSONA GENERATION:
- Personas MUST be labor-market segments (real candidate pools with distinct backgrounds and paths to this role), NOT personality archetypes ("The Results Driver", "The Relationship Builder")
- Each persona's archetype must reference their ACTUAL PRIOR BACKGROUND (e.g. "The Ad Ops Operator", "The SaaS CS Migrant", "The Recruitment Tech AM")
- Churn triggers and interview red flags MUST reflect the JD's hard constraints (shift, on-site, metrics depth, escalation pressure) — not generic dissatisfaction
- Sourcing channels MUST be market-specific (India: Naukri, Instahyre, LinkedIn India, AngelList/Wellfound, IIM Jobs, iimjobs.com; US: LinkedIn, Indeed, AngelList; UK: LinkedIn, CWJobs, TotalJobs)
- Currency and income figures MUST match the role's country (INR for India, USD for US, GBP for UK)
- The household income classification must use LOCAL market context — Indian HH income tiers differ fundamentally from US Pew tiers

INDIA HOUSEHOLD INCOME TIERS (use when role is India-based):
- Lower: <₹3L/yr — very high financial pressure
- Lower-middle: ₹3L–₹8L/yr — stretched, this role is a significant upgrade
- Middle: ₹8L–₹20L/yr — stable, motivated by career growth and brand name
- Upper-middle: ₹20L–₹40L/yr — selective, motivated by ownership and equity
- Upper: ₹40L+/yr — financially secure, motivated by impact and autonomy

STEP 1 — CLASSIFICATION AND PRESET ANCHORING
Classify the role into exactly one of five presets to anchor your persona count band:
- hourly_frontline (retail, warehouse, delivery, food service) → 5–6 personas
- gig_flexible (driver, tasker, platform, seasonal) → 5–6 personas
- licensed_skilled (nurse, CDL driver, electrician, certified trades) → 3–5 personas
- corporate_professional (marketer, analyst, PM, sales, HR, finance) → 3–4 personas
- executive_specialist (VP, director, principal engineer, C-suite) → 3 personas

STEP 2 — 5-AXIS SCORING
Score each axis exactly 1, 2, or 3 using these strict anchors. No decimals.
Axis A (Motivational Diversity): 1=Single Purpose | 2=Dual Track | 3=Fluid/Fragmented
Axis B (Age/Life Stage): 1=Single Band | 2=Dual Generation | 3=Omni-generational
Axis C (HH Income): 1=Homogeneous | 2=Bimodal Spread | 3=Full Spectrum
Axis D (Background/Education): 1=Rigid Gatekeeping | 2=Adjacency Friendly | 3=Zero Barriers
Axis E (Employment Context): 1=Single Status | 2=Hybrid Pool | 3=Gig/Volatile

Sum scores (5–15). Set persona count at the appropriate point within the preset band:
5–7=low end, 8–10=middle, 11–15=high end.

CROSS-AXIS VALIDATION: If Axis B = 3, Axis A CANNOT = 1.

STEP 3 — BRIDGE PERSONA EVALUATION
Include a Bridge Persona if AT LEAST 2 of these 4 signals are true:
1. Low barrier to entry (no degree/license, onboarding in days)
2. Flexible/short-term structure (contract, seasonal, part-time, gig)
3. Economic vulnerability (wages at or below local median, or hourly pay)
4. Broad applicant pool (accepts career changers, no niche experience required)
If triggered: bridge persona REPLACES the lowest-percentage non-essential segment. Never adds an extra persona beyond the score-dictated count.

STEP 4 — GENERATE PERSONAS WITH MAXIMUM VARIANCE
Maximize variance across personas on the highest-scoring axes. Personas that are minor demographic variations of each other are INVALID. Each must represent a genuinely distinct segment with different motivations, financial context, and life stage.

Pew HH Income Tiers (calibrate to local COL):
- Lower: <$35k — paycheck to paycheck, every dollar critical
- Lower-middle: $35k–$65k — stretched, gig income often essential not optional
- Middle: $65k–$100k — stable, gig work supplemental or chosen flexibility
- Upper-middle: $100k–$175k — comfortable, gig work genuinely optional
- Upper: $175k+ — financially secure, exploratory or bridge situation

SELF-VALIDATION — Before returning output, check all of these. If any fail, regenerate:
✗ REJECT if all personas share the same age range
✗ REJECT if all personas share the same sourcing channel
✗ REJECT if all personas share the same educational background
✗ REJECT if persona archetypes are personality types rather than prior-background segments
✗ REJECT if the JD's shift constraint, location, or required metrics do NOT appear in at least one churn_trigger or screening_question
✗ REJECT if income figures use the wrong currency for the role's country
✗ REJECT if sourcing channels are country-wrong (e.g. Indeed/Facebook Local for an India role)

Return ONLY valid JSON. No markdown. No explanation."""


def _build_prompt(jd_text: str, signals: dict, onet: dict, wages: dict, demos: str, fallback: bool) -> str:
    parts = [SYSTEM_PROMPT, "\n\n=== INPUT DATA ==="]

    if fallback:
        parts.append(f"\nFALLBACK MODE: URL extraction failed. Inferred title: {signals.get('title', 'Unknown')}. Generate personas from title + industry heuristics.")
    else:
        parts.append(f"\nCLEAN_JD_MARKDOWN:\n{jd_text[:4000]}")

    parts.append(f"\nSIGNALS EXTRACTED:\n{json.dumps(signals, indent=2)}")

    if onet:
        parts.append(f"\nONET_GROUNDING:\n{json.dumps(onet, indent=2)}")
    if wages:
        salary_str = (
            f"25th percentile: ${wages.get('percentile_25','?')}, "
            f"Median: ${wages.get('median','?')}, "
            f"75th percentile: ${wages.get('percentile_75','?')} "
            f"({wages.get('location','')}"
        )
        parts.append(f"\nMARKET_GROUNDING.salary_bounds: {salary_str}")
    if demos:
        parts.append(f"\nMARKET_GROUNDING.demographic_signals:\n{demos}")

    schema = '''
=== OUTPUT JSON SCHEMA (return ONLY this, no markdown) ===
{
  "jd_hard_filters": {
    "work_location": "string — exact city/country",
    "shift_constraint": "string — e.g. US shift 5PM-2AM IST, or null",
    "onsite_requirement": "mandatory|hybrid|remote",
    "experience_range": "string — e.g. 1-5 years",
    "preferred_industries": ["string"],
    "required_tools_metrics": ["string"],
    "explicit_disqualifiers": ["string"],
    "market_context": "India|US|UK|Global"
  },
  "role_summary": "string",
  "recruiter_brief": "string",
  "local_context": {
    "metro_area": "string",
    "median_hh_income": "string",
    "cost_of_living_index": "Low|Medium|High|Very High",
    "role_type": "gig|hourly|salaried|contract",
    "hiring_volume": "single-seat|moderate|high-volume"
  },
  "diversity_scoring": {
    "preset_used": "string",
    "axis_A": {"score": 1, "rationale": "string"},
    "axis_B": {"score": 1, "rationale": "string"},
    "axis_C": {"score": 1, "rationale": "string"},
    "axis_D": {"score": 1, "rationale": "string"},
    "axis_E": {"score": 1, "rationale": "string"},
    "total_score": 5,
    "target_persona_count": 3,
    "bridge_persona_plausible": false,
    "bridge_persona_included": false,
    "bridge_signals_present": 0
  },
  "personas": [
    {
      "metadata": {
        "name": "string",
        "archetype": "string — 3-word descriptor (The Gig Maximizer)",
        "segment_size_percentage": 30,
        "is_bridge_persona": false
      },
      "demographics": {
        "age_range": "string",
        "education": "string",
        "employment_status": "string"
      },
      "financials": {
        "pew_household_income_tier": "Lower|Lower-middle|Middle|Upper-middle|Upper",
        "household_income_range": "string",
        "household_income_note": "string — financial pressure implication for this persona",
        "target_monthly_income_from_role": "string",
        "hours_per_week_expected": "string",
        "income_dependency": "Primary|Secondary|Supplemental",
        "payment_preference": "Daily instant|Weekly|Monthly"
      },
      "drivers_and_friction": {
        "primary_motivation": "string — specific, not generic",
        "secondary_motivation": "string",
        "pain_point_1": "string",
        "pain_point_2": "string",
        "anti_pattern_signals": {
          "interview_red_flag": "string — what they say that signals 30-day quit risk",
          "churn_trigger": "string — single operational change causing ghosting"
        }
      },
      "tech_profile": {
        "tech_savviness_score": 3,
        "hardware_devices": ["string"],
        "key_apps": ["string"]
      },
      "evidence_confidence": {
        "overall_score": 80,
        "salary_confidence": "High|Medium|Low",
        "education_confidence": "High|Medium|Low",
        "demographic_confidence": "High|Medium|Low",
        "motivation_confidence": "Inferred",
        "notes": "string",
        "evidence_basis": ["string", "string"]
      },
      "screening_question": {
        "question": "string — recommended interview question",
        "why_it_matters": "string"
      },
      "recruiter_action": {
        "sourcing_channel": {
          "primary": "string — exact platform",
          "organic_play": "string — non-paid tactic"
        },
        "conversion_hook": {
          "headline": "string — exact ad headline",
          "core_value_prop": "string"
        },
        "funnel_friction_killer": "string — exact application process change"
      }
    }
  ],
  "job_ad_rewrite": {
    "current_jd_risk": "string",
    "missing_motivator": "string",
    "recommended_headline": "string",
    "bullet_to_add": "string",
    "bullet_to_remove": "string",
    "cta_improvement": "string"
  }
}'''
    parts.append(schema)
    return "\n".join(parts)


# ── Stage 5: LLM call (Gemini → Groq) ──────────────────────────────────────

def _call_gemini(prompt: str) -> str:
    """Call Gemini Flash via OpenAI-compatible endpoint."""
    resp = requests.post(
        "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
        headers={"Authorization": f"Bearer {GEMINI_KEY}", "Content-Type": "application/json"},
        json={
            "model": "gemini-1.5-flash",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": 6000,
            "response_format": {"type": "json_object"},
        },
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def _call_groq(prompt: str) -> str:
    """Call Groq Llama 3.3 70B."""
    resp = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {GROQ_KEY}", "Content-Type": "application/json"},
        json={
            "model": "llama-3.3-70b-versatile",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": 6000,
        },
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def _call_llm(prompt: str) -> str:
    """Gemini Flash → Groq fallback."""
    if GEMINI_KEY:
        try:
            return _call_gemini(prompt)
        except Exception as e:
            print(f"[nova] Gemini failed ({e}), falling back to Groq")
    if GROQ_KEY:
        return _call_groq(prompt)
    raise RuntimeError("No LLM key configured. Set GEMINI_API_KEY or GROQ_API_KEY.")


# ── Post-processing ─────────────────────────────────────────────────────────

def _normalize_segments(data: dict) -> dict:
    """Normalize segment_size_percentage values to sum exactly to 100."""
    personas = data.get("personas", [])
    if not personas:
        return data

    raw = [p.get("metadata", {}).get("segment_size_percentage", 0) for p in personas]
    total = sum(raw) or 1
    normalized = [round(v / total * 100) for v in raw]

    # Fix rounding residual on largest segment
    diff = 100 - sum(normalized)
    if diff != 0:
        max_idx = normalized.index(max(normalized))
        normalized[max_idx] += diff

    for i, p in enumerate(personas):
        p.setdefault("metadata", {})["segment_size_percentage"] = normalized[i]

    return data


def _parse_json(raw: str) -> dict:
    """Extract and parse JSON from LLM response."""
    # Try direct parse first
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    # Extract JSON block
    m = re.search(r"\{[\s\S]*\}", raw)
    if m:
        return json.loads(m.group())
    raise ValueError(f"No valid JSON in LLM response. Preview: {raw[:300]}")


# ── Main entry point ────────────────────────────────────────────────────────

def build_persona_response(text: str = "", url: str = "", source: str = "job_description") -> dict:
    """
    Main pipeline orchestrator.
    Args:
        text: raw job description text (if pasted)
        url:  job posting URL (if provided)
    Returns:
        Full persona payload dict matching PRD v2.1 schema
    """
    start = time.time()
    result = {"_pipeline": {}}

    # Stage 1: Fetch content
    jd_content = text.strip()
    fallback = False
    inferred_title = ""

    if not jd_content and url:
        fetch = _fetch_url(url)
        jd_content = fetch["content"]
        fallback = fetch["fallback_triggered"]
        inferred_title = fetch["inferred_title"]
        result["_pipeline"]["jina_chars"] = len(jd_content)
        result["_pipeline"]["fallback"] = fallback

    if not jd_content:
        raise ValueError("No job content provided and URL fetch returned empty content.")

    # Stage 2: Extract signals
    signals = _extract_signals(jd_content)
    if fallback and inferred_title:
        signals["title"] = signals.get("title") or inferred_title
    result["_pipeline"]["signals"] = signals

    # Stage 3: O*NET grounding
    onet = _onet_grounding(signals.get("title", ""))
    result["_pipeline"]["onet_soc"] = onet.get("soc_code", "")

    # Stage 4: CareerOneStop wages
    wages = _careeronestop_wages(signals.get("title", ""), signals.get("location", ""))
    result["_pipeline"]["wages"] = wages

    # Stage 5: Tavily demographics
    demos = _tavily_demographics(signals.get("title", ""), signals.get("location", ""))
    result["_pipeline"]["tavily_chars"] = len(demos)

    # Stage 6: Build prompt and call LLM
    prompt = _build_prompt(jd_content, signals, onet, wages, demos, fallback)
    raw = _call_llm(prompt)

    # Stage 7: Parse and normalize
    data = _parse_json(raw)
    data = _normalize_segments(data)

    # Metadata
    data["_provider"] = "gemini" if GEMINI_KEY else "groq"
    data["_pipeline_ms"] = round((time.time() - start) * 1000)

    return data
