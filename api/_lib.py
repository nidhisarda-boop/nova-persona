"""
Nova Candidate Map — v3 backend
Pipeline: Jina → O*NET → CareerOneStop → Tavily → Gemini/Groq → normalize
"""
import os, re, json, time, urllib.parse, socket, ipaddress
import requests


# ── Safety Layer ───────────────────────────────────────────────────────────

class SafetyError(ValueError):
    """Raised when input fails safety validation. Message is user-facing."""
    pass


class OutputValidationError(ValueError):
    """Raised when the LLM output fails schema validation. Not the user's fault —
    surfaced as a 502 so the frontend can ask the user to retry."""
    pass


# When false (default/production), internal pipeline fields are stripped from the
# response so they never reach the browser. Set DEBUG_PIPELINE=1 to expose them.
DEBUG_PIPELINE = os.environ.get("DEBUG_PIPELINE", "").lower() in ("1", "true", "yes", "on")


# Private/reserved IP ranges that must never be fetched (SSRF protection)
_PRIVATE_RANGES = [
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("169.254.0.0/16"),   # link-local
    ipaddress.ip_network("::1/128"),           # IPv6 loopback
    ipaddress.ip_network("fc00::/7"),          # IPv6 private
]

# Domains that commonly appear in spam/phishing/test abuse
_BLOCKED_DOMAIN_PATTERNS = [
    r"localhost", r"127\.\d+\.\d+\.\d+", r"0\.0\.0\.0",
    r"\.onion$", r"\.internal$", r"\.local$",
    r"burpcollaborator", r"ngrok\.io", r"requestbin",
    r"webhook\.site", r"pipedream\.net",
]

# Prompt injection patterns — sequences that try to hijack the LLM
_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions?",
    r"you\s+are\s+now\s+(a\s+)?(?:dan|jailbreak|unrestricted|evil)",
    r"(system|assistant|user)\s*:\s*",
    r"<\s*(system|instruction|prompt)\s*>",
    r"\[INST\]|\[\/INST\]",
    r"###\s*(instruction|system|override)",
    r"act\s+as\s+(if\s+you\s+are\s+)?(?:an?\s+)?(?:unrestricted|evil|jailbroken|unfiltered)",
    r"(disregard|forget|bypass|override)\s+(your\s+)?(training|guidelines|rules|safety|constraints)",
    r"(print|output|say|repeat|write|return)\s+.*?(password|token|secret|api.?key)",
    r"(execute|run|eval)\s*[\(\{]",
]

# Minimum signals that suggest content is job-related
_JOB_SIGNALS = [
    r"\b(job|role|position|vacancy|opening|opportunity|career|hiring|recruit)\b",
    r"\b(qualifications?|requirements?|responsibilities|experience|skills?|degree)\b",
    r"\b(salary|compensation|pay|benefits?|perks?|equity|bonus)\b",
    r"\b(apply|application|candidate|resume|cv|interview)\b",
    r"\b(full.?time|part.?time|remote|hybrid|on.?site|contract|freelance)\b",
    r"\b(company|team|department|manager|report|stakeholder)\b",
    r"\b(years?\s+of\s+experience|preferred|required|must.have|nice.to.have)\b",
]


def validate_url(url: str) -> str:
    """
    Validate and sanitize a URL before fetching.
    Returns the cleaned URL or raises SafetyError with a user-facing message.
    """
    url = url.strip()

    # Length check
    if len(url) > 2048:
        raise SafetyError("That URL is too long. Please paste a direct link to the job posting.")

    # Scheme check — only HTTP/HTTPS allowed
    try:
        parsed = urllib.parse.urlparse(url)
    except Exception:
        raise SafetyError("That doesn't look like a valid URL. Please paste a direct link to the job posting.")

    if parsed.scheme not in ("http", "https"):
        raise SafetyError(
            "Only https:// and http:// URLs are supported. "
            "File paths, FTP links, and other schemes are not accepted."
        )

    hostname = parsed.hostname or ""
    if not hostname:
        raise SafetyError("Could not read a hostname from that URL. Please check the link and try again.")

    # Block obviously bad domains
    for pattern in _BLOCKED_DOMAIN_PATTERNS:
        if re.search(pattern, hostname, re.IGNORECASE):
            raise SafetyError(
                "That URL doesn't appear to be a public job posting. "
                "Please paste a direct link to a publicly accessible job page."
            )

    # SSRF: resolve ALL addresses (IPv4 + IPv6) and reject if any is internal
    try:
        infos = socket.getaddrinfo(hostname, None)
        resolved = {info[4][0] for info in infos}
        for ip_str in resolved:
            ip_obj = ipaddress.ip_address(ip_str)
            # IPv4-mapped IPv6 (e.g. ::ffff:127.0.0.1) — unwrap to the IPv4 form
            if getattr(ip_obj, "ipv4_mapped", None):
                ip_obj = ip_obj.ipv4_mapped
            is_internal = (
                ip_obj.is_private or ip_obj.is_loopback or ip_obj.is_link_local
                or ip_obj.is_reserved or ip_obj.is_multicast or ip_obj.is_unspecified
                or any(ip_obj in r for r in _PRIVATE_RANGES)
            )
            if is_internal:
                raise SafetyError(
                    "That URL resolves to a private or internal address. "
                    "Please paste a link to a public job posting."
                )
    except SafetyError:
        raise
    except Exception:
        # DNS failure — let Jina handle it and fall back gracefully
        pass

    return url


def validate_text(text: str) -> str:
    """
    Validate pasted job description text.
    Returns sanitized text or raises SafetyError.
    """
    text = text.strip()

    # Length limits
    if len(text) < 30:
        raise SafetyError(
            "That text is too short to be a job description. "
            "Please paste the full job posting text."
        )
    if len(text) > 50_000:
        raise SafetyError(
            "That text is too long. Please paste the core job description "
            "(under 50,000 characters)."
        )

    # Repetition check — garbage like "aaaaaaa..." or "1111111..."
    if _is_repetitive_garbage(text):
        raise SafetyError(
            "That input doesn't look like a job description. "
            "Please paste the actual job posting text."
        )

    # Prompt injection check
    injection_hit = _check_injection(text)
    if injection_hit:
        raise SafetyError(
            "That input contains content Nova can't process safely. "
            "Please paste a standard job description."
        )

    # Must have at least 2 job-related signals
    signals_found = sum(
        1 for pattern in _JOB_SIGNALS
        if re.search(pattern, text, re.IGNORECASE)
    )
    if signals_found < 2:
        raise SafetyError(
            "That doesn't appear to be a job description. "
            "Nova works best with actual job postings. "
            "Please paste the role requirements, responsibilities, and qualifications."
        )

    return text


def sanitize_for_prompt(text: str) -> str:
    """
    Strip or neutralize prompt injection sequences before embedding in LLM prompt.
    Does NOT raise — silently cleans the content.
    """
    # Remove common injection trigger sequences
    for pattern in _INJECTION_PATTERNS:
        text = re.sub(pattern, "[removed]", text, flags=re.IGNORECASE)

    # Strip unusual control characters but preserve normal whitespace
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    # Cap at 8000 chars for prompt safety (truncate with notice)
    if len(text) > 8000:
        text = text[:8000] + "\n[content truncated for safety]"

    return text


def _is_repetitive_garbage(text: str) -> bool:
    """Detect strings that are mostly repeated characters (spam/garbage)."""
    if len(text) < 50:
        return False
    # Count unique chars relative to length
    unique_ratio = len(set(text.lower())) / len(text)
    if unique_ratio < 0.02:
        return True
    # Check for very long runs of the same character
    if re.search(r"(.)\1{49,}", text):
        return True
    return False


def _check_injection(text: str) -> str | None:
    """Return the matched injection pattern string, or None if clean."""
    for pattern in _INJECTION_PATTERNS:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            return m.group(0)
    return None

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

GIG/CONTRACTOR ROLE SPECIAL RULES (apply when preset = gig_flexible or hourly_frontline):
1. MINIMUM AGE: Do NOT generate personas below the minimum eligible age for the role. For rideshare/delivery platforms: minimum is typically 21 (Lyft, Uber) or 18 in select markets. Never assume 18 unless confirmed by JD or platform requirements page. Use "21–25 (verify local requirements)" not "18–25".
2. EARNINGS CLAIMS: Do NOT use specific earnings claims like "Earn up to $X/week" or "Make $X per hour" unless these are directly quoted from the JD. Platforms like Lyft have faced FTC enforcement for inflated earnings claims. Instead use: "Earn on your schedule", "See pay upfront", "Cash out fast" — benefit-led, not amount-specific.
3. VEHICLE ACCESS SEGMENTATION: For vehicle-required gig roles, segment candidates by vehicle situation:
   - "Owns qualifying vehicle" — straightforward to start
   - "Needs rental/Express Drive option" — high intent, blocked by upfront cost, convert via rental program
   - "Unsure if vehicle qualifies" — needs vehicle requirements clarification first
   This MUST appear in at least one persona's onboarding friction or churn trigger.
4. ONBOARDING QUESTIONS NOT INTERVIEW QUESTIONS: For contractor/gig roles, replace "screening_question" with an onboarding/conversion question — the goal is to predict activation and retention, not assess fit for employment. Example: "What would make you prioritize Lyft over your other earning apps this week?" not "Tell me about yourself."
5. ANTI-REPETITION: The following MUST be DIFFERENT across all personas:
   - primary_motivation (ban: using "flexibility and autonomy" for more than one persona)
   - sourcing_channel.primary (no two personas can share the same primary channel)
   - conversion_hook.headline (every headline must be genuinely different)
   - churn_trigger (cannot be the same payment/income issue for every persona)
6. COMPETING APPS: For platform gig roles, the multi-app segment is real and large. At least one persona should address a driver who already uses competing platforms (Uber, DoorDash, Instacart). Their acquisition hook is "marginal value of adding Lyft" not "join Lyft."
7. ROLE TYPE: For contractor roles, use "contractor" in employment_status fields, not "employee" or "part-time worker".

SELF-VALIDATION — Before returning output, check all of these. If any fail, regenerate:
✗ REJECT if all personas share the same age range
✗ REJECT if all personas share the same sourcing channel
✗ REJECT if all personas share the same educational background
✗ REJECT if persona archetypes are personality types rather than prior-background segments
✗ REJECT if the JD's shift constraint, location, or required metrics do NOT appear in at least one churn_trigger or screening_question
✗ REJECT if income figures use the wrong currency for the role's country
✗ REJECT if sourcing channels are country-wrong (e.g. Indeed/Facebook Local for an India role)
✗ REJECT if any persona age range starts below the role's minimum eligibility age
✗ REJECT if any conversion_hook.headline contains a specific earnings dollar/currency amount not sourced directly from the JD
✗ REJECT if all primary_motivations are variations of "flexibility and autonomy"
✗ REJECT if two or more personas share the same sourcing_channel.primary
✗ REJECT if gig role has no persona addressing vehicle access barrier or competing platform users

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
            "response_format": {"type": "json_object"},
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


# Top-level keys the frontend depends on
_REQUIRED_TOP_KEYS = [
    "jd_hard_filters", "role_summary", "recruiter_brief",
    "personas", "job_ad_rewrite",
]
_MAX_PERSONAS = 10


def _scan_for_injection(value, _depth=0):
    """Recursively walk the output and raise if any string field echoes a prompt
    injection sequence (e.g. the model parroting 'ignore previous instructions'
    or '<system>' back into a persona name). Guards the browser from rendering
    attacker-controlled control text."""
    if _depth > 12:  # defensive against pathological nesting
        return
    if isinstance(value, str):
        for pat in _INJECTION_PATTERNS:
            if re.search(pat, value, re.IGNORECASE):
                raise OutputValidationError(
                    "The generated persona failed a safety check. Please try again."
                )
    elif isinstance(value, dict):
        for v in value.values():
            _scan_for_injection(v, _depth + 1)
    elif isinstance(value, list):
        for v in value:
            _scan_for_injection(v, _depth + 1)


def validate_output(data: dict) -> dict:
    """Schema-validate the LLM persona payload before it reaches the browser.
    Raises OutputValidationError on anything malformed, missing, or unsafe.
    Clamps out-of-range numerics in place rather than failing the whole request."""
    if not isinstance(data, dict):
        raise OutputValidationError("Model returned a non-object response.")

    missing = [k for k in _REQUIRED_TOP_KEYS if k not in data]
    if missing:
        raise OutputValidationError(
            f"Model response was missing required fields: {', '.join(missing)}."
        )

    personas = data.get("personas")
    if not isinstance(personas, list) or not personas:
        raise OutputValidationError("Model response contained no personas.")
    if len(personas) > _MAX_PERSONAS:
        raise OutputValidationError("Model returned an implausible number of personas.")

    for p in personas:
        if not isinstance(p, dict):
            raise OutputValidationError("Model returned a malformed persona entry.")
        meta = p.get("metadata")
        if not isinstance(meta, dict):
            raise OutputValidationError("Persona is missing its metadata block.")
        name = meta.get("name")
        if not isinstance(name, str) or not name.strip():
            raise OutputValidationError("Persona is missing a name.")
        # Clamp segment size to a sane range rather than trusting the model
        seg = meta.get("segment_size_percentage", 0)
        if not isinstance(seg, (int, float)) or isinstance(seg, bool) or not (0 <= seg <= 100):
            meta["segment_size_percentage"] = 0

    # Reject any injection text the model may have echoed into output strings
    _scan_for_injection(data)
    return data


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

    # ── Safety: validate inputs before doing anything ──────────────────────
    if url:
        url = validate_url(url)          # raises SafetyError if bad URL
    if text:
        text = validate_text(text)       # raises SafetyError if garbage/injection

    # Stage 1: Fetch content
    jd_content = text.strip() if text else ""
    fallback = False
    inferred_title = ""

    if not jd_content and url:
        fetch = _fetch_url(url)
        jd_content = fetch["content"]
        fallback = fetch["fallback_triggered"]
        inferred_title = fetch["inferred_title"]
        result["_pipeline"]["jina_chars"] = len(jd_content)
        result["_pipeline"]["fallback"] = fallback

        # Validate fetched content too (could be login wall, captcha page, or injected page)
        if jd_content and not fallback:
            try:
                validate_text(jd_content)
            except SafetyError:
                # Page loaded but isn't a job description — trigger fallback
                fallback = True
                jd_content = inferred_title or ""
                result["_pipeline"]["fallback_reason"] = "fetched_page_not_job_related"

    if not jd_content and not inferred_title:
        raise ValueError("No job content provided and URL fetch returned empty content.")

    # ── Sanitize before injecting into LLM prompt ──────────────────────────
    jd_content = sanitize_for_prompt(jd_content)

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

    # Stage 7: Parse, validate, and normalize
    data = _parse_json(raw)
    data = validate_output(data)      # schema + safety check before it leaves the server
    data = _normalize_segments(data)

    # Debug internals — only exposed when DEBUG_PIPELINE is enabled.
    if DEBUG_PIPELINE:
        data["_provider"] = "gemini" if GEMINI_KEY else "groq"
        data["_pipeline_ms"] = round((time.time() - start) * 1000)
        data["_pipeline"] = result["_pipeline"]
        return data

    # Production: strip every internal/debug field so nothing leaks to the browser.
    return {k: v for k, v in data.items() if not k.startswith("_")}
