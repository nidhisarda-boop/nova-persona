"""
Nova Candidate Map — v3 backend
Pipeline: Jina → O*NET → CareerOneStop → Tavily → Gemini/Groq → normalize
"""
import os, re, json, time, urllib.parse, socket, ipaddress, html
import requests


# ── Safety Layer ───────────────────────────────────────────────────────────

class SafetyError(ValueError):
    """Raised when input fails safety validation. Message is user-facing."""
    pass


class OutputValidationError(ValueError):
    """Raised when the LLM output fails schema validation. Not the user's fault —
    surfaced as a 502 so the frontend can ask the user to retry."""
    pass


class LLMUnavailableError(RuntimeError):
    """Raised when every configured LLM provider is rate-limited or unavailable
    (HTTP 429/5xx, timeouts). Surfaced as a 503 with a user-facing message."""
    pass


class SearchPageError(Exception):
    """Raised when a URL is a job-search/category results page, not a single
    posting. Surfaced as a 200 with page_type=search_results + a mode chooser,
    NOT an error — the user picks: market map / pick a job / paste JD."""
    def __init__(self, role_hint: str = "", market: str = "Global"):
        self.role_hint = role_hint
        self.market = market
        super().__init__("search_results_page")


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
    """Detect strings that are mostly repeated characters (spam/garbage).

    Uses the ABSOLUTE number of distinct characters, not a ratio: real text
    always has dozens of distinct characters regardless of length, while
    garbage like "aaaa..." or "12121212..." has very few. (A ratio-based
    check wrongly rejects any legitimate text longer than ~2,500 chars.)"""
    if len(text) < 50:
        return False
    # Very few distinct characters across the whole input → garbage.
    if len(set(text.lower())) < 10:
        return True
    # Mostly non-letters (numeric dumps, code, ID lists, symbol junk).
    # Uses Unicode-aware isalpha so non-Latin scripts (e.g. Hindi JDs) count.
    if sum(c.isalpha() for c in text) / len(text) < 0.35:
        return True
    # A very long run of a single repeated character → garbage.
    if re.search(r"(.)\1{49,}", text):
        return True
    return False


def _is_invalid_title(title: str) -> bool:
    """True if an inferred title is unusable for persona generation — empty or
    containing no alphabetic characters (e.g. a numeric job ID like 8574881002).
    Single real words ('Driver', 'Bartender') are intentionally allowed."""
    if not title or not title.strip():
        return True
    return not any(c.isalpha() for c in title)


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
# Extra Gemini keys (from SEPARATE Google projects = independent free quota).
# Add GEMINI_API_KEY_2, _3, … to multiply free headroom before any paid fallback.
GEMINI_KEYS = [k for k in [
    GEMINI_KEY,
    os.environ.get("GEMINI_API_KEY_2", ""),
    os.environ.get("GEMINI_API_KEY_3", ""),
    os.environ.get("GEMINI_API_KEY_4", ""),
] if k]
GROQ_KEY        = os.environ.get("GROQ_API_KEY", "")
CEREBRAS_KEY    = os.environ.get("CEREBRAS_API_KEY", "")
OPENROUTER_KEY  = os.environ.get("OPENROUTER_API_KEY", "")
# GitHub Models accepts a GitHub PAT; allow either env name.
GITHUB_MODELS_KEY = os.environ.get("GITHUB_MODELS_TOKEN") or os.environ.get("GITHUB_TOKEN", "")
# Additional LLM providers (extra API keys)
OPENAI_KEY      = os.environ.get("OPENAI_API_KEY", "")
ANTHROPIC_KEY   = os.environ.get("ANTHROPIC_API_KEY", "")
MISTRAL_KEY     = os.environ.get("MISTRAL_API_KEY", "")
TOGETHER_KEY    = os.environ.get("TOGETHER_API_KEY", "")
XAI_KEY         = os.environ.get("XAI_API_KEY", "")
NVIDIA_KEY      = os.environ.get("NVIDIA_NIM_API_KEY", "")
SAMBANOVA_KEY   = os.environ.get("SAMBANOVA_API_KEY", "")
COHERE_KEY      = os.environ.get("COHERE_API_KEY") or os.environ.get("COHERSE_API_KEY", "")
ZHIPU_KEY       = os.environ.get("ZHIPU_API_KEY", "")
SILICONFLOW_KEY = os.environ.get("SILICONFLOW_API_KEY", "")
DEEPSEEK_KEY    = os.environ.get("DEEPSEEK_API_KEY", "")

# Data enrichment (real salary grounding via Adzuna — global coverage).
ENABLE_ENRICHMENT = os.environ.get("ENABLE_DATA_ENRICHMENT", "").lower() in ("1", "true", "yes", "on")
ADZUNA_APP_ID  = os.environ.get("ADZUNA_APP_ID", "")
ADZUNA_APP_KEY = os.environ.get("ADZUNA_APP_KEY", "")
FRED_KEY       = os.environ.get("FRED_API_KEY", "")

# LLM output ceiling. Default 8000 fits gemini-2.0-flash's 8192 cap. To allow
# fuller 6-persona maps, set GEMINI_MODEL=gemini-2.5-flash and LLM_MAX_TOKENS=16000.
GEMINI_MODEL   = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
LLM_MAX_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", "16000"))

TIMEOUT = 20
# Best-effort enrichment (Adzuna/FRED) must never blow the function time budget.
# The fallback chain can issue several sequential calls, so each one is kept short.
ENRICH_TIMEOUT = int(os.environ.get("ENRICH_TIMEOUT", "6"))
# Per-LLM-provider request timeout. MUST be well under the Vercel function
# maxDuration (60s) so one slow provider can't consume the whole budget and get
# the function killed before the fallback chain or our error JSON can return.
LLM_TIMEOUT = int(os.environ.get("LLM_TIMEOUT", "120"))
# Early/fast providers fail FAST (a hung one must not burn the budget); only the
# FINAL last-resort provider (usually Claude, which is slower to generate a full
# map) gets the generous remaining window.
LLM_TIMEOUT_FAST = int(os.environ.get("LLM_TIMEOUT_FAST", "18"))
# Hard wall-clock deadline for the whole request (seconds). Set at pipeline start;
# the LLM loop stops starting new providers once we're within one LLM_TIMEOUT of it,
# guaranteeing we return our own JSON (honest error) instead of a 504.
REQUEST_BUDGET_S = int(os.environ.get("REQUEST_BUDGET_S", "285"))
_REQUEST_DEADLINE = 0.0


# ── Stage 1: Jina content fetch ────────────────────────────────────────────

def _extract_title_from_url(url: str) -> str:
    """Fallback: pull a job title from the URL slug. Returns '' if the slug has
    no real words (e.g. a numeric job ID like .../jobs/8574881002)."""
    try:
        path = urllib.parse.urlparse(url).path
        slug = path.strip("/").split("/")[-1]
        title = re.sub(r"[-_]", " ", slug).strip()
        title = re.sub(r"\b(job|jobs|career|careers|apply|posting|opening)\b", "", title, flags=re.I).strip()
        # Reject slugs with no real alphabetic words (pure numeric IDs, etc.)
        if not re.search(r"[A-Za-z]{2,}", title):
            return ""
        return title.title()
    except Exception:
        return ""


def _extract_jina_title(content: str) -> str:
    """Jina Reader prefixes its output with a 'Title: ...' line — pull it out.
    This is a far more reliable role title than the URL slug for sites like
    Greenhouse/Lever/Workday whose URLs are just numeric IDs."""
    m = re.match(r"\s*Title:\s*(.+)", content or "")
    if not m:
        return ""
    title = m.group(1).strip()
    # Drop trailing site-name noise after a separator (e.g. "Global Brand Manager | AB InBev")
    title = re.split(r"\s[|–\-]\s", title)[0].strip()
    return title if re.search(r"[A-Za-z]{2,}", title) else ""


_SEARCH_URL_PATTERNS = [
    r"ziprecruiter\.com/Jobs/",          # ZipRecruiter category/search pages
    r"indeed\.[^/]+/jobs\?",             # Indeed search
    r"linkedin\.com/jobs/search",
    r"/jobs/search\b", r"/job-search\b", r"/jobs/browse\b", r"/browse-jobs\b",
    r"/jobs/?$", r"/search\b",
    r"[?&](q|k|keyword|keywords|search|query)=",
]


def _looks_like_search_page(url: str, content: str) -> bool:
    """Detect a job-search / category results page (many jobs) vs a single posting."""
    u = url or ""
    for pat in _SEARCH_URL_PATTERNS:
        if re.search(pat, u, re.IGNORECASE):
            return True
    # Content signal: a listing page shows many distinct pay figures + many "apply"s.
    if content:
        pays = len(re.findall(r"[$£€₹]\s?\d", content))
        applies = len(re.findall(r"\bapply\b", content, re.IGNORECASE))
        titles = len(re.findall(r"\b(per hour|/hr|an hour)\b", content, re.IGNORECASE))
        if pays >= 6 and (applies >= 5 or titles >= 4):
            return True
    return False


def _role_cluster_from_url(url: str) -> str:
    """Pull a role-category label from a search/category URL — the title-cased
    slug (ziprecruiter /Jobs/Blue-Collar-Worker) or the q=/keyword= query."""
    try:
        parsed = urllib.parse.urlparse(url or "")
    except Exception:
        return ""
    m = re.search(r"/Jobs?/([^/?#]+)", url or "", re.IGNORECASE)
    if m:
        slug = re.sub(r"[-_+]+", " ", m.group(1)).strip()
        slug = re.sub(r"\b(jobs?|careers?|hiring|near me)\b", "", slug, flags=re.IGNORECASE).strip()
        if re.search(r"[A-Za-z]{2,}", slug):
            return slug.title()
    q = urllib.parse.parse_qs(parsed.query)
    for k in ("q", "keyword", "keywords", "query", "search"):
        if q.get(k) and q[k][0].strip():
            return urllib.parse.unquote_plus(q[k][0]).strip()
    return ""


def _greenhouse_ids(url: str):
    """Pull (board_token, job_id) from a Greenhouse URL, else (None, None)."""
    m = re.search(r"greenhouse\.io/(?:embed/job_app\?for=)?([\w-]+)/jobs/(\d+)", url)
    if m:
        return m.group(1), m.group(2)
    m = re.search(r"greenhouse\.io/embed/job_app\?for=([\w-]+).*?[?&](?:token|gh_jid)=(\d+)", url)
    if m:
        return m.group(1), m.group(2)
    return None, None


def _fetch_greenhouse(board: str, job_id: str) -> str:
    """Fetch a Greenhouse posting via its public board API — returns clean full JD
    text (the human page is a JS shell Jina cannot read)."""
    r = requests.get(
        f"https://boards-api.greenhouse.io/v1/boards/{board}/jobs/{job_id}",
        headers={"Accept": "application/json"},
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    d = r.json()
    title = d.get("title", "") or ""
    loc = ((d.get("location") or {}).get("name") or "").strip()
    body = d.get("content", "") or ""
    # Greenhouse double-escapes HTML; unescape twice, then strip tags.
    body = html.unescape(html.unescape(body))
    body = re.sub(r"<\s*(br|/p|/li|/h\d)\s*>", "\n", body, flags=re.IGNORECASE)
    body = re.sub(r"<[^>]+>", " ", body)
    body = re.sub(r"[ \t]+", " ", body)
    body = re.sub(r"\n[ \t]*\n+", "\n\n", body).strip()
    header = f"Title: {title}\n"
    if loc:
        header += f"Work Location: {loc}\n"
    return f"{header}\n{body}".strip()


def _fetch_url(url: str) -> dict:
    """Fetch job posting. Prefers ATS-native APIs (Greenhouse) for clean structured
    content, falls back to Jina Reader. Returns {content, fallback_triggered, inferred_title}."""
    # ATS-specific: Greenhouse public API returns the full, clean JD as JSON.
    board, job_id = _greenhouse_ids(url)
    if board and job_id:
        try:
            gh = _fetch_greenhouse(board, job_id)
            if len(gh) >= 200:
                return {
                    "content": gh,
                    "fallback_triggered": False,
                    "inferred_title": _extract_jina_title(gh) or _extract_title_from_url(url),
                }
        except Exception:
            pass  # fall through to Jina

    content = ""
    for attempt in (1, 2):
        try:
            headers = {"Accept": "text/plain"}
            if JINA_KEY:
                headers["Authorization"] = f"Bearer {JINA_KEY}"
            resp = requests.get(f"https://r.jina.ai/{url}", headers=headers, timeout=TIMEOUT)
            content = resp.text if resp.status_code == 200 else ""
        except Exception:
            content = ""
        if len(content) >= 200:
            break
        if attempt == 1:
            time.sleep(1)  # brief retry for transient Jina throttling

    # Prefer the page's own title (from Jina) over the URL slug.
    inferred = _extract_jina_title(content) or _extract_title_from_url(url)

    return {
        "content": content,
        "fallback_triggered": len(content) < 200,
        "inferred_title": inferred,
    }


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


# ── Stage 3b: Adzuna real salary grounding (GLOBAL — US/UK/India/EU/CA/AU) ──

_ADZUNA_COUNTRIES = {"US": "us", "UK": "gb", "India": "in", "Canada": "ca", "Australia": "au"}
_ADZUNA_EU = [
    (r"netherland|nederland|amsterdam|rotterdam|the hague|utrecht|eindhoven", "nl"),
    (r"german|deutschland|berlin|munich|frankfurt|hamburg", "de"),
    (r"\bfrance\b|paris|lyon|marseille", "fr"),
    (r"\bspain\b|madrid|barcelona|valencia", "es"),
    (r"\bitaly\b|milan|rome|turin", "it"),
    (r"belgium|brussels|antwerp", "be"),
    (r"poland|warsaw|krakow", "pl"),
    (r"austria|vienna", "at"),
]


def _adzuna_country(market: str, blob: str) -> str:
    """Map a detected market (and content) to an Adzuna country code, or ''."""
    if market in _ADZUNA_COUNTRIES:
        return _ADZUNA_COUNTRIES[market]
    if market == "EU":
        for pat, cc in _ADZUNA_EU:
            if re.search(pat, blob, re.IGNORECASE):
                return cc
    return ""


def _title_from_summary(summary: str) -> str:
    """Pull a clean job title out of the LLM's role_summary sentence, e.g.
    'The Shift Leader at Whataburger in San Antonio will…' -> 'Shift Leader'.
    Used as a fallback when deterministic title extraction came up empty."""
    if not summary:
        return ""
    m = re.match(
        r"\s*(?:The|A|An)?\s*([A-Z][\w/&.+-]*(?:\s+[\w/&.+-]+){0,5}?)\s+"
        r"(?:at|in|is|are|will|would|role\b|position\b|responsible|oversees?|"
        r"manages?|leads?|handles?|drives?|supports?)\b",
        summary,
    )
    if not m:
        return ""
    title = re.sub(r"\s+", " ", m.group(1)).strip(" .,-")
    # Reject if it grabbed a whole clause or obvious non-title
    if len(title.split()) > 6 or len(title) < 3:
        return ""
    return title


_US_STATE_ABBR = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas",
    "CA": "California", "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware",
    "FL": "Florida", "GA": "Georgia", "HI": "Hawaii", "ID": "Idaho",
    "IL": "Illinois", "IN": "Indiana", "IA": "Iowa", "KS": "Kansas",
    "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine", "MD": "Maryland",
    "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi",
    "MO": "Missouri", "MT": "Montana", "NE": "Nebraska", "NV": "Nevada",
    "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico", "NY": "New York",
    "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma",
    "OR": "Oregon", "PA": "Pennsylvania", "RI": "Rhode Island", "SC": "South Carolina",
    "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas", "UT": "Utah",
    "VT": "Vermont", "VA": "Virginia", "WA": "Washington", "WV": "West Virginia",
    "WI": "Wisconsin", "WY": "Wyoming", "DC": "District of Columbia",
}

# Major US cities -> state (covers the metros in _MARKET_SIGNALS["US"])
_US_CITY_STATE = {
    "san antonio": "Texas", "dallas": "Texas", "houston": "Texas", "fort worth": "Texas",
    "austin": "Texas", "el paso": "Texas", "phoenix": "Arizona", "tucson": "Arizona",
    "los angeles": "California", "san francisco": "California", "san diego": "California",
    "san jose": "California", "fresno": "California", "sacramento": "California",
    "san bernardino": "California", "new york": "New York", "chicago": "Illinois",
    "seattle": "Washington", "boston": "Massachusetts", "philadelphia": "Pennsylvania",
    "pittsburgh": "Pennsylvania", "jacksonville": "Florida", "miami": "Florida",
    "tampa": "Florida", "orlando": "Florida", "columbus": "Ohio", "cleveland": "Ohio",
    "cincinnati": "Ohio", "charlotte": "North Carolina", "raleigh": "North Carolina",
    "indianapolis": "Indiana", "denver": "Colorado", "nashville": "Tennessee",
    "memphis": "Tennessee", "oklahoma city": "Oklahoma", "las vegas": "Nevada",
    "detroit": "Michigan", "portland": "Oregon", "atlanta": "Georgia",
    "baltimore": "Maryland", "milwaukee": "Wisconsin", "albuquerque": "New Mexico",
    "kansas city": "Missouri", "omaha": "Nebraska", "st louis": "Missouri",
    "st. louis": "Missouri", "minneapolis": "Minnesota", "new orleans": "Louisiana",
    "salt lake city": "Utah", "washington": "District of Columbia",
}


def _us_state_for(location: str, content: str) -> str:
    """Resolve a US state name from the location string or JD content, for an
    Adzuna state-level fallback. Returns '' if none found."""
    loc = location or ""
    blob = f"{loc}\n{(content or '')[:1500]}"
    # "City, TX" abbreviation (prefer the location field)
    m = re.search(r",\s*([A-Z]{2})\b", loc) or re.search(r",\s*([A-Z]{2})\b", blob)
    if m and m.group(1) in _US_STATE_ABBR:
        return _US_STATE_ABBR[m.group(1)]
    # Full state name anywhere in the blob
    for st in _US_STATE_ABBR.values():
        if re.search(rf"\b{re.escape(st)}\b", blob, re.IGNORECASE):
            return st
    # City lookup
    key = loc.strip().lower().split(",")[0].strip()
    return _US_CITY_STATE.get(key, "")


_LAST_ADZUNA_DIAG = {}  # TEMP DEBUG: records why the last Adzuna call did/didn't fire

# Titles too generic to query Adzuna on their own — they match unrelated, often
# higher-paying jobs across other industries and pollute the average.
_GENERIC_TITLE_RX = re.compile(
    r"^\s*(?:team\s*member|crew\s*member|crew|team\s*lead(?:er)?|team|member|"
    r"associate|staff(?:\s*member)?|worker|general\s*worker|hourly\s*associate|"
    r"floor\s*staff|operative)\s*$",
    re.IGNORECASE,
)

# Industry keyword to disambiguate a generic title, detected from the JD content.
_INDUSTRY_HINTS = [
    (r"restaurant|fast.?food|kitchen|barista|qsr|drive.?thru|food\s*service|"
     r"cook|fryer|burger|menu|guest|crew|whataburger|mcdonald|wendy|taco bell", "restaurant"),
    (r"\bretail\b|store|merchandis|planogram|cashier|stockroom|shopper|checkout", "retail"),
    (r"warehouse|fulfil|distribution|forklift|pallet|pick.?pack|loading dock|sortation", "warehouse"),
    (r"hotel|hospitality|housekeep|front desk|guest room|resort|concierge", "hospitality"),
    (r"call\s*center|contact\s*center|customer support|inbound|outbound calls", "call center"),
    (r"caregiv|home care|patient|nursing|\bcna\b|\baide\b|assisted living", "care"),
    (r"cleaning|janitorial|custodial", "cleaning"),
    (r"delivery|courier|\bdriver\b|route", "delivery"),
    (r"warehouse|manufactur|assembly line|production line|plant", "manufacturing"),
]


def _industry_hint(content: str) -> str:
    blob = (content or "")[:2500]
    for rx, kw in _INDUSTRY_HINTS:
        if re.search(rx, blob, re.IGNORECASE):
            return kw
    return ""


def _enrich_salary(title: str, location: str, market: str, content: str) -> dict:
    """Real salary benchmark from Adzuna for THIS title × location. Global coverage.
    Returns {} unless ENABLE_DATA_ENRICHMENT and Adzuna keys are set."""
    global _LAST_ADZUNA_DIAG
    _LAST_ADZUNA_DIAG = {
        "enrichment_enabled": bool(ENABLE_ENRICHMENT),
        "app_id_present": bool(ADZUNA_APP_ID),
        "app_key_present": bool(ADZUNA_APP_KEY),
        "title": title or "",
        "market": market or "",
    }
    if not (ENABLE_ENRICHMENT and ADZUNA_APP_ID and ADZUNA_APP_KEY and title):
        _LAST_ADZUNA_DIAG["reason"] = "gated_off (enrichment flag, missing app_id/app_key, or empty title)"
        return {}
    cc = _adzuna_country(market, f"{location}\n{content[:1500]}")
    _LAST_ADZUNA_DIAG["country_code"] = cc
    if not cc:
        _LAST_ADZUNA_DIAG["reason"] = f"no_adzuna_country_for_market ({market})"
        return {}

    # Qualify over-generic titles ("Team Member", "Associate", "Crew") with an
    # industry keyword from the JD so Adzuna matches the RIGHT jobs instead of
    # averaging in unrelated higher-paying roles that share the word.
    what_term = title
    if _GENERIC_TITLE_RX.match(title or ""):
        hint = _industry_hint(content)
        if hint:
            what_term = f"{title} {hint}"
            _LAST_ADZUNA_DIAG["qualified_query"] = what_term

    # Attempt chain — narrow to broad: city -> state -> national. Adzuna's
    # city-level salary coverage is patchy, so we widen the geography step by
    # step rather than jumping straight to the country average.
    attempts = []
    if location:
        attempts.append((location, "city"))
    if cc == "us":
        state = _us_state_for(location, content)
        if state and state.lower() != (location or "").strip().lower():
            attempts.append((state, "state"))
    attempts.append((None, "national"))

    last_reason = "no_mean_in_response (no salary data for this title×location)"
    for where, geo in attempts:
        try:
            params = {
                "app_id": ADZUNA_APP_ID, "app_key": ADZUNA_APP_KEY,
                "what": what_term, "results_per_page": 50, "content-type": "application/json",
            }
            if where:
                params["where"] = where
            r = requests.get(
                f"https://api.adzuna.com/v1/api/jobs/{cc}/search/1", params=params, timeout=ENRICH_TIMEOUT
            )
            _LAST_ADZUNA_DIAG["http_status"] = r.status_code
            _LAST_ADZUNA_DIAG["geo"] = geo
            if r.status_code != 200:
                last_reason = f"http_{r.status_code} (likely bad credentials if 401/403)"
                _LAST_ADZUNA_DIAG["reason"] = last_reason
                continue
            d = r.json()
            mean = d.get("mean")
            _LAST_ADZUNA_DIAG["count"] = d.get("count", 0)
            if not mean:
                last_reason = f"no_mean ({geo} query had no salaried postings)"
                _LAST_ADZUNA_DIAG["reason"] = last_reason
                continue
            sals = [x.get("salary_min") for x in d.get("results", []) if x.get("salary_min")]
            sals += [x.get("salary_max") for x in d.get("results", []) if x.get("salary_max")]
            out = {"source": "Adzuna", "country": cc.upper(), "mean": round(mean),
                   "count": d.get("count", 0), "geo": geo, "geo_name": where or ""}
            if sals:
                lo, hi = round(min(sals)), round(max(sals))
                # Adzuna's `mean` is the full-dataset average, but low/high come from
                # the page-1 results sample, which may not bracket it. Clamp so the
                # displayed range ALWAYS contains the average — never "avg > max".
                out["low"], out["high"] = min(lo, out["mean"]), max(hi, out["mean"])
            _LAST_ADZUNA_DIAG["reason"] = f"ok ({geo})"
            return out
        except Exception as e:
            last_reason = f"exception: {type(e).__name__}"
            _LAST_ADZUNA_DIAG["reason"] = last_reason
            continue
    return {}


def _enrich_income(market: str) -> dict:
    """Real US median household income from FRED (anchors the income tiers).
    US-only for now; returns {} unless enrichment + FRED key are set."""
    if not (ENABLE_ENRICHMENT and FRED_KEY and market == "US"):
        return {}
    try:
        r = requests.get(
            "https://api.stlouisfed.org/fred/series/observations",
            params={"series_id": "MEHOINUSA672N", "api_key": FRED_KEY,
                    "file_type": "json", "sort_order": "desc", "limit": 1},
            timeout=ENRICH_TIMEOUT,
        )
        if r.status_code != 200:
            return {}
        obs = (r.json().get("observations") or [{}])[0]
        val = obs.get("value")
        if not val or val == ".":
            return {}
        return {"source": "FRED", "us_median_household_income": round(float(val)),
                "year": (obs.get("date") or "")[:4]}
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
    """Deterministically extract structured fields from JD text. These are handed
    to the model as authoritative — far more reliable than asking a weak model to
    re-read a raw blob for salary/location."""
    signals = {"title": "", "company": "", "location": "", "salary": "",
               "arrangement": "", "experience": "", "responsibilities": []}

    # Title: look for common patterns
    title_patterns = [
        r"^\s*Title:\s*([A-Za-z][^\n]{2,70})",                 # Jina/Greenhouse title line
        r"(?:job title|position)\s*:\s*([A-Za-z][^\n,|]{3,60})",
        r"\brole\s*:\s*([A-Za-z][^\n,|]{3,60})",               # requires a colon (not "Role Overview")
        r"^#+\s*([A-Za-z][^\n]{3,60})$",
    ]
    for pat in title_patterns:
        m = re.search(pat, text[:500], re.MULTILINE | re.IGNORECASE)
        if m:
            signals["title"] = m.group(1).strip()
            break

    # Location — prefer an explicitly labeled line, else first "City, ST/Country"
    loc_label = re.search(
        r"(?:work location|location|based in|office)\s*[:\-]\s*"
        r"([A-Z][A-Za-z]+(?:(?:\s*/\s*|\s)[A-Z][A-Za-z]+){0,4}(?:,\s*[A-Z][A-Za-z]+(?:\s[A-Z][A-Za-z]+)?)?)",
        text, re.IGNORECASE,
    )
    if loc_label:
        loc = loc_label.group(1).strip().rstrip(".").strip()
        loc = re.sub(r"\s+(area|region|metro\w*|and|with|the)\b.*$", "", loc, flags=re.IGNORECASE).strip()
        signals["location"] = loc
    else:
        loc_m = re.search(
            r"\b([A-Z][a-z]+(?:[ /][A-Z][a-z]+){0,3},\s*(?:[A-Z]{2}\b|[A-Z][a-z]+(?:\s[A-Z][a-z]+)?))",
            text[:1500],
        )
        if loc_m:
            signals["location"] = loc_m.group(1)

    # Salary — labeled line first (handles "$165,000 - $185,000", "$165.000", "+ bonus")
    sal_label = re.search(
        r"(?:salary|compensation|pay|base(?:\s*salary)?)\s*[:\-]?\s*"
        r"([\$£€₹]\s?[\d.,]+(?:\s*[-–]\s*[\$£€₹]?\s?[\d.,]+)?(?:\s*[KkMm])?(?:\s*\+\s*[A-Za-z ]+)?)",
        text, re.IGNORECASE,
    )
    if sal_label:
        signals["salary"] = sal_label.group(1).strip()
    else:
        # Unlabeled $-figure fallback. Be careful NOT to mistake a business metric
        # (e.g. "own a $4M+ location", "$2.5M in annual sales") for someone's pay.
        # Reject million-scale figures (a bare $XM is essentially never an individual
        # base salary) and any figure sitting in a revenue/asset/volume context.
        _BIZ_CTX = re.compile(
            r"\b(?:location|locations|restaurant|store|stores|outlet|outlets|unit|units|"
            r"sales|revenue|budget|portfolio|valuation|funding|raised?|turnover|aum|auv|"
            r"p&l|facility|facilities|account|accounts|deal|deals|in\s+annual|in\s+revenue|"
            r"in\s+sales)\b",
            re.IGNORECASE,
        )
        chosen = ""
        for mt in re.finditer(
            r"[\$£€₹]\s?[\d.,]+(?:\s*[-–]\s*[\$£€₹]?\s?[\d.,]+)?\s*([KkMm])?", text
        ):
            suffix = (mt.group(1) or "").lower()
            after = text[mt.end():mt.end() + 30]
            if suffix == "m":            # $4M / $2.5M → business figure, not a salary
                continue
            if _BIZ_CTX.search(after):   # "$X ... location/sales/revenue" → not a salary
                continue
            chosen = mt.group(0).strip()
            break
        if chosen:
            signals["salary"] = chosen
        else:
            lpa = re.search(r"[\d.,]+\s*(?:LPA|lakhs?|crore)\b", text, re.IGNORECASE)
            if lpa:
                signals["salary"] = lpa.group(0).strip()

    # Experience requirement (English "years" or Dutch "jaar")
    exp_m = re.search(
        r"(\d+\+?(?:\s*[-–]\s*\d+)?\s*(?:years?|jaar)(?:\s*(?:of\s*experience|ervaring|relevante))?)",
        text, re.IGNORECASE,
    )
    if exp_m:
        signals["experience"] = exp_m.group(1).strip()

    # Arrangement
    if re.search(r"\bon.?site\b|\bin.?office\b|\bin.?person\b", text, re.IGNORECASE):
        signals["arrangement"] = "on-site"
    elif re.search(r"\bhybrid\b", text, re.IGNORECASE):
        signals["arrangement"] = "hybrid"
    elif re.search(r"\bremote\b|work from home|\bwfh\b", text, re.IGNORECASE):
        signals["arrangement"] = "remote"

    # Responsibility bullets — pull lines under "What You'll Do" / "Responsibilities"
    resp_block = re.search(
        r"(?:what you.?ll do|responsibilities|role overview|key responsibilities)\b[:\s]*"
        r"(.+?)(?:what you.?ll need|what we offer|requirements|qualifications|benefits|$)",
        text, re.IGNORECASE | re.DOTALL,
    )
    if resp_block:
        bullets = []
        for line in resp_block.group(1).splitlines():
            line = line.strip(" -•*\t•")
            if 12 <= len(line) <= 220 and re.search(r"[a-z]", line):
                bullets.append(line)
        signals["responsibilities"] = bullets[:8]

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

STEP 0.5 — COMPENSATION & LOCATION ANCHOR (the JD overrides all benchmark data)
- If the JD states a salary or salary range, that IS the primary compensation anchor. Derive household_income_range, target_monthly_income_from_role (gross monthly ≈ annual ÷ 12), and the income tier from it. Do NOT replace it with national/median benchmark numbers.
- If the JD names a specific city/metro, use THAT as metro_area and calibrate cost_of_living_index to that city (e.g. New York, San Francisco = Very High). Never collapse a specific city into a national aggregate.
- Any MARKET_GROUNDING wage data provided below is a SECONDARY benchmark only. It must never override an explicit JD salary or a JD-stated location.

STEP 0.6 — OPERATING CENTER (classify before generating personas)
Identify the role's true operating center — exactly one of: Strategy-heavy | Execution-heavy | Sales/client-heavy | Analytics-heavy | Creative/production-heavy | Operations-heavy.
Every persona MUST map to talent pools that fit this operating center as defined by the JD's ACTUAL responsibilities. Do NOT over-index on generic brand strategy, MBA ambition, or market research unless the JD explicitly emphasizes them. Example: if a JD centers on campaign execution, creative operations, media planning, and agency management, personas must come from campaign-execution, creative-ops, media/social, and agency-operator pools — not "market research analyst" or "MBA generalist".

CRITICAL RULES FOR PERSONA GENERATION:
- EVIDENCE INTEGRITY: evidence_basis, notes, and confidence fields may ONLY cite a source whose data was actually supplied in INPUT DATA (STRUCTURED_JD, ONET_GROUNDING, MARKET_GROUNDING). NEVER invent source names such as "LinkedIn postings", "Glassdoor salary data", "Indeed insights", or "alumni placement data".
- STRUCTURED_JD fields COUNT AS SOURCED. When salary or location came from STRUCTURED_JD, treat them as grounded: evidence_basis should say "Sourced from the job posting", salary/location are High confidence, and sourced_vs_inferred must state which facts came from the JD vs were inferred. Only when neither the JD nor an external source provided salary/location do you write "Inferred from JD requirements; external salary source not used" and keep overall_score ≤ 65.
- NO NAMED SALARY SOURCES: never attribute an inferred salary/income figure to a specific website, report, or dataset you were not given (e.g. "SalaryExpert", "Payscale", "Glassdoor", "BLS", "ZipRecruiter salary data"). If a figure is inferred, write exactly "inferred from US labor-market norms" — naming a source you did not receive is fabrication. EXCEPTION: if ADZUNA_SALARY data is present in INPUT DATA, it WAS actually queried for this role — you MUST use it to ground compensation, cite it as "Adzuna market data", and mark salary as SOURCED / High confidence.
- NO FABRICATED PERKS OR OWNERSHIP: never claim the role offers equity, ownership, stock, profit-share, a franchise, or any specific benefit the JD does not explicitly state — in conversion_hook, job_ad_rewrite, recommendations, or anywhere. In particular do NOT use the word "equity" or imply ownership unless the JD literally states it. If the JD mentions an advancement path (e.g. "fast-track to Operating Partner"), describe it as a "path to [role] and higher earning potential", NOT as equity or ownership.
- Personas MUST be labor-market segments (real candidate pools with distinct backgrounds and paths to this role), NOT personality archetypes ("The Results Driver", "The Relationship Builder")
- Each persona's archetype AND name must name a SOURCEABLE prior-background talent pool — a real group you could filter for on LinkedIn/Naukri by their last role, industry, or employer type (e.g. "Agency Campaign Operator", "CPG Brand Marketer", "Media-Agency Social Lead", "The Ad Ops Operator", "The SaaS CS Migrant")
- BANNED as an archetype or name: functional skills or personality traits that EVERY candidate for this role would share — e.g. "Data-Driven", "Creative", "Storyteller", "Data-Driven Brand Builder", "Creative Storyteller", "Strategic Thinker", "Results-Driven", "Innovative". These are not distinct candidate pools and cannot be sourced. Litmus test: if you cannot rephrase the archetype as "their last role was X" or "they come from Y industry", it is INVALID — fix it before returning.
- Churn triggers and interview red flags MUST reflect the JD's hard constraints (shift, on-site, metrics depth, escalation pressure) — not generic dissatisfaction
- Sourcing channels MUST be market-specific (India: Naukri, Instahyre, LinkedIn India, AngelList/Wellfound, IIM Jobs, iimjobs.com; US: LinkedIn, Indeed, AngelList; UK: LinkedIn, CWJobs, TotalJobs)
- Currency and income figures MUST match the role's country (INR for India, USD for US, GBP for UK)
- The household income classification must use LOCAL market context — Indian HH income tiers differ fundamentally from US Pew tiers
- household_income_range is the TOTAL HOUSEHOLD income (not just this role's pay). Calibrate it to each segment's life stage and earner structure, vary it WIDELY across personas, and MATCH THE BANDS TO THE ROLE'S TIER:
  • ENTRY-LEVEL / HOURLY / FRONTLINE / GIG roles: a young first-job worker, student, or single earner sits LOW ($18k–$40k); a working parent or dual-earner household taking this job FOR the income is modest ($40k–$75k). HARD LOGIC: any persona who NEEDS this income — a "working parent", a "second job" / supplemental earner, a single earner, a re-entry worker — by definition does NOT live in a six-figure household; someone in a $100k+ household does not take a fast-food job to make ends meet. So such personas MUST sit ≤ $75k. The displaced/overqualified BRIDGE may be the HIGHEST household in the set, but for an hourly role it caps around $70k–$110k (they are in a reduced/transitional situation, possibly with a working partner — NOT at their former management peak). NO persona on an hourly crew map may exceed ~$110k; never output a $120k+ household here.
  • CORPORATE / PROFESSIONAL / EXECUTIVE roles: household income is naturally HIGHER and should reflect the role's salary plus likely dual income — six-figure households (and for senior/executive roles, multi-six-figure) are normal and expected. Do NOT suppress these or force them into hourly bands.
  Either way: do NOT collapse household income to just this role's salary, and do NOT make every persona's range identical.

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

Sum scores (5–15) and set the persona count strictly from the score within the preset band:
- 3–4 band (corporate_professional): score ≤7 → 3 personas; score ≥8 → 4 personas.
- 3–5 band (licensed_skilled): score ≤7 → 3; 8–11 → 4; ≥12 → 5.
- 5–6 band (gig/hourly): score ≤9 → 5; ≥10 → 6.
- 3 band (executive_specialist): always 3.
GENERATE EXACTLY this many persona objects, and set target_persona_count to the same number. The personas array length MUST equal target_persona_count.

CROSS-AXIS VALIDATION: If Axis B = 3, Axis A CANNOT = 1.

SENIORITY FLOOR — MATCH EVERY PERSONA TO THE ROLE'S ACTUAL LEVEL (applies before all persona generation below):
Determine the role's seniority from its title + requirements. If it is an EXPERIENCED or MANAGEMENT-level role — i.e. it manages people / owns a P&L / location, or requires multiple years of experience (e.g. Restaurant Manager, Store Manager, General Manager, Engineering Manager) — then EVERY persona, INCLUDING the bridge, must be someone realistically qualified for THIS role TODAY or a near-ready internal promotion.
- The LARGEST direct-fit pool is people already doing this job or exactly one clear step below — e.g. for a Restaurant Manager: current Restaurant Managers and Assistant/General Managers at competitors. NOT shift-level crew, and NOT entry-level workers.
- Do NOT include an entry-level, crew, trainee, "aspirant", or "low-barrier entry" pool as a persona for a management role. That pool belongs in a Shift Leader / management-trainee map, not here. If you are tempted to add it, replace it with a qualified adjacent pool instead (e.g. retail STORE manager, hospitality OPERATIONS manager, multi-unit supervisor).
- For a management role the bridge MUST be a comparably-senior ADJACENT pool (retail store manager, hospitality operations manager, veteran operations/logistics leader), ~10–20% of the pool — never an under-qualified entry pool, and never over-indexed.

STEP 3 — BRIDGE PERSONA EVALUATION
First apply the SENIORITY FLOOR above. The 4 low-barrier signals below select an ENTRY-LEVEL bridge and apply ONLY to genuinely entry-level / hourly-frontline roles (cashier, crew member, driver, warehouse associate). For any experienced/management role, IGNORE these 4 signals and use the comparably-senior adjacent bridge from the SENIORITY FLOOR rule instead.
Include an entry-level Bridge Persona ONLY for an entry-level role and if AT LEAST 2 of these 4 signals are true:
1. Low barrier to entry (no degree/license, onboarding in days)
2. Flexible/short-term structure (contract, seasonal, part-time, gig)
3. Economic vulnerability (wages at or below local median, or hourly pay)
4. Broad applicant pool (accepts career changers, no niche experience required)
If triggered: the bridge persona OCCUPIES ONE of the score-dictated persona slots — it does NOT change the total count. Still generate EXACTLY the score-dictated number of personas; one of them is simply the bridge. The bridge should be a MINORITY of the pool (roughly 15–25% segment_size_percentage), not co-equal with the core pools.

CORPORATE / SENIOR BRIDGE (for corporate_professional and executive_specialist presets, where the 4 signals above rarely apply): include a bridge persona when a displaced senior-talent pool is plausible — e.g. an experienced director or manager from the same domain recently affected by layoffs or restructuring who would take this role as an immediate landing spot. They bring strong capability but carry elevated flight risk. Make that risk explicit in their anti_pattern_signals.churn_trigger (e.g. "leaves the moment a director-level seat opens elsewhere"). This bridge is ONE of the score-dictated personas (it does not reduce the count) and should be ~15–25% of the pool.

OVERQUALIFIED BRIDGE (a senior/manager-level person applying to a role well BELOW their level — e.g. a displaced restaurant manager applying to an hourly Team Member job): this is realistic but it is a SMALL, high-churn stop-gap pool. Keep it to roughly 5–10% of the pool (NOT 15–25%), and make the overqualification + high flight risk explicit in their churn_trigger ("leaves the moment a management seat opens"). On their household_income_range, the number MUST be COHERENT with their narrative — choose ONE and make it consistent:
  (a) REDUCED / between-roles household — they need this income now; household sits in the lower-middle to middle band, and the role is framed as primary income; OR
  (b) AFFLUENT dual-income household — a working partner or savings makes this a SUPPLEMENTAL stop-gap; a higher (even six-figure) household is fine HERE, but you MUST then frame the role explicitly as SECONDARY/supplemental income, state the household's main income comes from elsewhere, and keep their expected_monthly modest.
  NEVER pair a high household income with a narrative that implies this hourly role is their primary income — the figure and the story must match. The other personas remain the core qualified pools.

STEP 4 — GENERATE PERSONAS WITH MAXIMUM VARIANCE
Maximize variance across personas on the highest-scoring axes. Personas that are minor demographic variations of each other are INVALID. Each must represent a genuinely distinct segment with different motivations, financial context, and life stage.
DISTINCTNESS TEST (apply to every pair of personas): two personas are DUPLICATES — and you MUST merge them or replace one — if they share the same LIFE STAGE + EXPERIENCE LEVEL + PRIMARY MOTIVATION, even under different names. "Food Service Enthusiast" and "Career Starter" are the SAME entry-level segment; "Career Changer" and "Bridge Worker" are usually the SAME segment. Names alone do not make personas distinct — the underlying pool must differ.
FRONTLINE / HOURLY ROLES — pick from genuinely different LIFE-STAGE pools, not near-duplicate entry-level labels. A strong set draws from: (a) FIRST-JOB youth (16–20, no prior work history, wants a first stable paycheck); (b) STUDENT part-timer (needs shifts that flex around class); (c) EXPERIENCED QSR/retail worker switching FROM a named competitor (McDonald's, Taco Bell, Burger King) for better pay or schedule; (d) SECOND-JOB / working-parent income supporter (older, wants reliable predictable hours and weekly pay); (e) RE-ENTRY / career-changer (gap in work history, returning to the workforce). Choose the pools that fit THIS role and market; do NOT output two personas that are both "entry-level enthusiast/starter."

Pew HH Income Tiers — US roles (calibrate to local COL):
- Lower: <$35k — paycheck to paycheck, every dollar critical
- Lower-middle: $35k–$65k — stretched, gig income often essential not optional
- Middle: $65k–$100k — stable, gig work supplemental or chosen flexibility
- Upper-middle: $100k–$175k — comfortable, gig work genuinely optional
- Upper: $175k+ — financially secure, exploratory or bridge situation

EU / Eurozone HH Income Tiers (use when DETECTED_MARKET is EU; Western-EU/Netherlands calibration):
- Lower: <€30k/yr
- Lower-middle: €30k–€50k
- Middle: €50k–€75k
- Upper-middle: €75k–€120k
- Upper: €120k+/yr
Note: a single earner taking home €5k–€8k/month (~€60k–€96k/yr) is Middle to Upper-middle in this scale, NOT "Lower-middle". Calibrate the tier to the actual figure.

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
   - primary_motivation (ban: using "flexibility and autonomy" for more than one persona; also do NOT reuse "job security" or "career advancement" as the primary for more than one)
   - pain_points (the friction/frustration list cannot be the same set — e.g. "poor work-life balance, limited opportunities" — repeated across personas)
   - sourcing_channel.primary (no two personas can share the same primary channel)
   - conversion_hook.headline (every headline must be genuinely different)
   - churn_trigger (cannot be the same payment/income issue for every persona)
6. COMPETING APPS: For platform gig roles, the multi-app segment is real and large. At least one persona should address a driver who already uses competing platforms (Uber, DoorDash, Instacart). Their acquisition hook is "marginal value of adding Lyft" not "join Lyft."
7. ROLE TYPE: For contractor roles, use "contractor" in employment_status fields, not "employee" or "part-time worker".

SELF-VALIDATION — Before returning output, check all of these. If any fail, regenerate:
✗ REJECT if all personas share the same age range
✗ REJECT if all personas share the same sourcing channel
✗ REJECT if any persona is an entry-level / crew / trainee / "aspirant" pool for a role that manages people, owns a P&L/location, or requires multiple years of experience — every persona (incl. the bridge) must be qualified-today or a near-ready internal promotion for THIS role's level
✗ REJECT if is_bridge_persona is true for a CORE / direct-fit pool (first-job, student, experienced crew, working parent) or for more than one persona — only the single overqualified/displaced/adjacent bridge carries that flag
✗ REJECT if target_monthly_income_from_role is not a dollar figure (it MUST be money, e.g. "$2,200"; never a dependency word like "Primary"/"Secondary"/"Supplemental")
✗ REJECT if any persona's metadata.name is a personal/human name (e.g. "Alex Rivera", "Samantha Lee") — names MUST be descriptive cohort/segment labels, never invented people
✗ REJECT if any two personas are the SAME segment under different names — i.e. they share the same life stage AND experience level AND primary motivation (e.g. two entry-level "starter/enthusiast" pools, or a "career changer" and a "bridge worker" that describe the same person). Merge them and replace the freed slot with a genuinely distinct pool, or reduce the persona count.
✗ REJECT if a persona's employment_status / label says "part-time" but hours_per_week_expected is 30+ (part-time means under 30 hrs/week) — make the label and the hours consistent.
✗ REJECT if jd_hard_filters or persona_jd_mismatch.you_want lists a certification, license, or credential (e.g. ServSafe / food-safety certification) that the JD does NOT actually require — do not invent hard requirements the posting never stated. If the JD marks a cert as optional / "may vary" / "where applicable", frame it as "willing to follow food-safety standards and complete required certification where applicable", NOT as a credential the candidate must already hold.
✗ REJECT if, for an entry-level/hourly role, more than one persona has a six-figure ($100k+) household_income_range — household income must skew lower and vary; six figures is rare and never clustered.
✗ REJECT if all personas share the same educational background
✗ REJECT if persona archetypes are personality types rather than prior-background segments
✗ REJECT if any archetype/name is a functional skill or trait ("Data-Driven", "Creative Storyteller", "Strategic", "Brand Builder") instead of a sourceable prior-background pool
✗ REJECT if STRUCTURED_JD provides a work_location or salary but you output "Not specified" / omit it — you MUST use the structured value verbatim
✗ REJECT if pew_household_income_tier contradicts household_income_range per the tier bands (e.g. labeling ~$80k "Lower-middle" when that band is $35k–$65k)
✗ REJECT if household_income_range is IDENTICAL across personas — each segment has a different household financial reality (a dual-income veteran, a single-parent earner, a student household, and a career-changer do NOT all sit in the same band). Give each persona a household_income_range that reflects its own life stage and earner structure; no two personas may share the exact same range, and the spread across personas must be meaningful (the lowest and highest segments should differ by at least one full tier).
✗ REJECT if target_monthly_income_from_role is inconsistent with the JD's stated salary (gross monthly ≈ JD annual ÷ 12) when the JD states a salary
✗ REJECT if overall_score is ≥ 75 while salary AND location were NOT grounded (no structured salary/location and no external source) — in that case overall_score must be ≤ 65
✗ REJECT if the JD's shift constraint, location, or required metrics do NOT appear in at least one churn_trigger or screening_question
✗ REJECT if income figures use the wrong currency for the role's country
✗ REJECT if sourcing channels are country-wrong (e.g. Indeed/Facebook Local for an India role)
✗ REJECT if any persona age range starts below the role's minimum eligibility age
✗ REJECT if any conversion_hook.headline contains a specific earnings dollar/currency amount not sourced directly from the JD
✗ REJECT if Axis C = 1 while personas span multiple income tiers or mix single/dual-income households (that is bimodal → Axis C ≥ 2)
✗ REJECT if Axis D = 1 while personas come from clearly different industries/entry-paths (→ Axis D ≥ 2)
✗ REJECT if any persona's sourcing_channel lacks a real boolean search_string or has fewer than 5 target_companies
✗ REJECT if any persona is missing application_dropoff_risk.risk and .fix
✗ REJECT if two personas share the same conversion_hook.headline
✗ REJECT if deal_breakers are identical across personas or merely restate the JD's hard requirements (they must be the CANDIDATE's segment-specific objections)
✗ REJECT if all primary_motivations are variations of "flexibility and autonomy"
✗ REJECT if two or more personas share the same sourcing_channel.primary
✗ REJECT if gig role has no persona addressing vehicle access barrier or competing platform users

ACTIVATION OUTPUT PRIORITY (this is the core product — spend your effort here):
The buyer is a Talent Marketing / Employer Brand lead. They need: WHO to target, WHERE to find them, WHAT message converts them, and WHY they drop off. Make these fields excellent:
- sourcing_channel.search_string: a REAL copy-paste boolean string (role-title synonyms AND industry terms AND skill phrases). Not "search LinkedIn".
- sourcing_channel.target_companies: 5-8 named companies/talent pools to poach this exact persona from (competitors, adjacent-industry leaders).
- conversion_hook: a DIFFERENT, persona-specific pitch per persona — speak to that pool's specific motivation.
- application_dropoff_risk: why this segment ghosts or declines, and the precise fix. This is more valuable than any interview question.
- job_ad_rewrite.recommended_headline: candidate-facing and specific to the role's true operating center (e.g. "Lead global campaign execution for an iconic beer portfolio", not "Lead Global Brand Strategy").
The screening_question / diagnostic is SECONDARY — keep it to one strong question; do not let it compensate for weak sourcing or segmentation.

AXIS SCORE ↔ PERSONA CONSISTENCY (score the axes to MATCH the pools you actually generate):
- Axis C (HH Income): if your personas span more than one income tier, or mix single-income and dual-income households, or cover a household-income range wider than ~$40k, Axis C is at least 2 (bimodal); a full spectrum is 3. Do NOT score Axis C = 1 while generating personas with visibly different income structures.
- Axis D (Background): if personas come from different industries or entry paths (e.g. CPG + tech/SaaS + agency), that is adjacency-friendly — Axis D is at least 2.
- Axis A (Motivation): if personas have distinctly different primary_motivations, Axis A is at least 2.
Re-check your 5-axis scores against the final persona set before returning; a senior role at a large global company is rarely a uniform 1-1-1 pool.

CONDITIONAL FIELDS (avoid no-signal filler):
- hours_per_week_expected, payment_preference (financials) and tech_savviness_score, hardware_devices (tech_profile) carry signal ONLY for gig_flexible / hourly_frontline / licensed_skilled frontline roles. For corporate_professional and executive_specialist presets these are constant noise — set them to null. Do NOT invent laptop/phone models or a "5/5" tech score for office roles.
- key_apps stays for all roles (it informs sourcing — where the segment actually spends time).

FIELD RICHNESS & HONESTY (V2):
- tech_profile.hardware_devices and key_apps: give specific, realistic examples for THIS segment (e.g. a digital marketer: "Figma", "Google Analytics 4", "Slack", "Asana"; a field gig worker: "budget Android phone", "Google Maps", "the platform app"). Treat these as ILLUSTRATIVE inferences, never as verified facts, and never build evidence claims on them.
- household_income_note: explain the financial pressure or motivation implication for this segment (e.g. "moving from volatile agency bonuses to a predictable corporate base; motivated by stability and 401(k) match more than raw base") — do not merely restate the number.
- screening_question: the high_risk_answer and risk_rationale must be concrete and role-specific, so a recruiter with no training can evaluate the answer.
- recruiter_action: sourcing and conversion must be concrete and executable — specific platforms, example search filters, and example employer names as ILLUSTRATIONS — consistent with the role's market and currency.
- PROOFREAD everything before returning. Correct spelling, correct brand and product names, no typos or garbled words. The job_ad_rewrite.recommended_headline in particular must be clean, correct, and free of errors.

PERSONA QUALIFICATION (pools must actually qualify for the role):
- MINIMUM EXPERIENCE: every persona must plausibly meet the JD's stated minimum experience. If the JD requires 8+ years (or "8 jaar"), do NOT create an early-career / junior persona implying fewer years.
- EXPERIENCE IS NOT AGE: a years-of-experience requirement constrains seniority, NOT age — someone can have 3+ (even 8+) years by their late 20s/30s. Vary age ranges realistically and WIDELY across personas (e.g. 26–34, 32–45, 45–58). Do NOT cluster every persona in a near-retirement (50+) band. If all personas end up 50+, you have made an error — fix it.
- NO OVERQUALIFIED OR EXOTIC POOLS: do not invent segments who would not realistically apply for this role — e.g. a PhD academic, a hobbyist/DIY homeowner, or a far-senior executive for a hands-on field/technician/hourly job. Every persona must be someone who would plausibly take THIS role.
- INDUSTRY-NATIVE POOL: if the JD explicitly requires experience in a specific industry or function (e.g. staffing / recruitment / detachering, fintech, healthcare), at least one persona MUST come natively from that industry — that is usually the single best-fit pool, so prioritize it over generic adjacent pools.
- COMPETITOR-TALENT FIRST (skilled trades / experienced roles): order the pools by fit. (1) people already doing this exact job, including at named competitors, are the LARGEST persona at roughly 45–55% of the pool; (2) adjacent field-service / route-based trades next, smaller (e.g. HVAC, appliance/pool service, facilities, lawn/landscaping applicators); (3) then a small bridge (~15–20%). Do NOT lead with niche or exotic pools, and do not let an adjacent pool exceed the direct-experience pool.

OUTPUT POLISH & HONESTY (V2.1 — enterprise quality):
- ONE PERSONA PER DISTINCT POOL: if the JD describes distinct functional areas (e.g. campaign execution, creative studio / content operations, media planning, agency management), give each its own persona rather than merging them. A creative-studio/content-ops pool is distinct from a media-planning pool and from an agency-campaign-lead pool — if the JD covers all of them, prefer the upper end of the persona band so each is represented.
- NEVER INVENT COMPANY POLICY: drop-off fixes must not assert benefits or policies the JD does not state (remote work, comp, perks). If unknown, recommend CLARIFYING in the JD — e.g. "Clarify onsite expectations and any flexibility for global-coordination calls", NOT "Add that remote days are permitted."
- NO UNSUPPORTED BRAND NAMES: do not name specific brands/products (e.g. Budweiser) unless they appear in the JD. Otherwise say "AB InBev's portfolio" or "the brand."
- PLAIN-ENGLISH EVIDENCE: evidence_basis must use human-readable source descriptions ("the job posting", "US labor-market norms"). NEVER emit internal variable names (MARKET_GROUNDING, STRUCTURED_JD, ONET_GROUNDING) — those are not sources.
- PROOFREAD: put a space between a number and its unit ("1 billion", not "1billion"). Never emit placeholders like "(truncated…)". When quoting a JD line to remove, quote it in full or paraphrase it — never a fragment.
- COMPENSATION LABELING: if the JD discloses NO pay/salary structure, do NOT assume "salaried" and do NOT invent one. Set local_context.role_type to the actual employment type stated (e.g. "full-time"; "contract"/"hourly" only if stated) and local_context.posted_compensation to null. Any household-income or target-pay figure in that case is INFERRED — say so in household_income_note and keep evidence confidence modest.
- JD-NATIVE CONVERSION HOOKS: pull the JD's strongest concrete selling points into conversion_hook and job_ad_rewrite where the JD states them — e.g. "dispatch from home", "company vehicle", "year-round work", "401(k) match", named certifications. A direct-fit persona's hook should lead with the JD's best perks, not generic phrasing.
- NO UNSUPPORTED OPERATIONAL CLAIMS: never invent specific operational promises absent from the JD — no "onboarding begins within 48 hours", "instant enrollment", guaranteed timelines, or exact pay amounts not in the JD. CTAs stay benefit-led, not fabricated-specific.

V1.5 INTELLIGENCE FIELDS (populate well — these are the product's wedge):
- deal_breakers: concrete hard no's for THIS segment, specific to the JD (not generic platitudes).
- candidate_journey: where THIS segment drops at each funnel stage (discovery → click → apply → interview → offer → early-churn). One crisp, segment-specific reason per stage.
- evidence_confidence.confidence_breakdown: honestly bucket fields into high/medium/low. JD-sourced facts (location, salary, experience) are HIGH; inferred demographics (household income, age range) are LOW. Be honest — this is the trust feature.
- recruiter_action.outreach_script: a real, ready-to-send 3-sentence message in the segment's tone, referencing the JD's best perks. No fabricated specifics.
- persona_jd_mismatch (top-level): "who you want" vs "who the JD will actually attract", the gap, and the fix. Nova's wedge is WHO WILL ACTUALLY APPLY — not the company's ideal. Make this sharp and employer-brand-useful.
- CALIBRATE REQUIREMENTS TO THE ROLE'S ACTUAL SCOPE: do not inflate seniority beyond what the role needs. A manager of ONE high-volume location is NOT a multi-unit/area manager — never state "multi-unit management experience" (or similar broader scope) as a hard requirement for a single-location role. If the JD mentions broader scope only as a future growth path, treat it as "preferred", not "required". In jd_hard_filters and persona_jd_mismatch.you_want, frame advanced/broader experience as "preferred" and keep the hard minimum tied to the role's real scope (e.g. "2+ years managing a high-volume restaurant; multi-unit exposure preferred"), so you don't screen out strong single-unit candidates.

Return ONLY valid JSON. No markdown. No explanation."""


_MARKET_SIGNALS = {
    "India": [
        r"\bindia\b", r"\bindian\b",
        r"\b(?:bangalore|bengaluru|mumbai|delhi|gurgaon|gurugram|hyderabad|pune|"
        r"chennai|kolkata|noida|ahmedabad|jaipur|kochi|chandigarh|indore)\b",
        r"₹", r"\bINR\b", r"\b(?:lakh|lakhs|lpa|crore)\b", r"\brs\.?\s?\d",
    ],
    "UK": [
        r"\bunited kingdom\b", r"\bu\.?k\.?\b",
        r"\b(?:london|manchester|birmingham|edinburgh|glasgow|leeds|bristol)\b",
        r"£", r"\bGBP\b", r"\bper annum\b",
    ],
    "EU": [
        r"€", r"\bEUR\b", r"\bZZP\b", r"\bjaar\b", r"\bper maand\b",
        r"\b(?:netherlands|nederland|germany|deutschland|france|spain|espa|italy|"
        r"belgium|ireland|austria|portugal|sweden|denmark|finland|poland)\b",
        r"\b(?:amsterdam|rotterdam|the hague|den haag|utrecht|eindhoven|berlin|munich|"
        r"frankfurt|paris|madrid|barcelona|milan|rome|brussels|dublin|lisbon)\b",
        r"\b(?:detacher|uitzend|personeelbemiddeling|werkervaring)\w*",
    ],
    "Canada": [
        r"\bcanada\b", r"\bcanadian\b", r"\bCAD\b", r"\bC\$",
        r"\b(?:toronto|vancouver|montreal|calgary|ottawa|edmonton|mississauga)\b",
    ],
    "Australia": [
        r"\baustralia\b", r"\baustralian\b", r"\bAUD\b", r"\bA\$", r"\bsuperannuation\b",
        r"\b(?:sydney|melbourne|brisbane|perth|canberra|adelaide)\b",
    ],
    "US": [
        r"\bunited states\b", r"\bu\.?s\.?a?\.?\b", r"\b401\(?k\)?\b", r"\bUSD\b",
        r"\bserv\s?safe\b", r"\bOSHA\b", r"\bGED\b", r"\bH-?1B\b", r"\bEEO\b",
        # Major US metros (broad coverage, not just tech hubs)
        r"\b(?:new york|san francisco|seattle|austin|boston|chicago|los angeles|"
        r"san antonio|dallas|houston|fort worth|phoenix|philadelphia|san diego|"
        r"san jose|jacksonville|columbus|charlotte|indianapolis|denver|nashville|"
        r"oklahoma city|el paso|las vegas|detroit|memphis|portland|miami|atlanta|"
        r"baltimore|milwaukee|albuquerque|tucson|fresno|sacramento|kansas city|"
        r"omaha|raleigh|tampa|orlando|cleveland|cincinnati|pittsburgh|st\.?\s?louis|"
        r"minneapolis|new orleans|salt lake city|san bernardino)\b",
        # US state names
        r"\b(?:alabama|alaska|arizona|arkansas|california|colorado|connecticut|"
        r"delaware|florida|georgia|hawaii|idaho|illinois|indiana|iowa|kansas|"
        r"kentucky|louisiana|maryland|massachusetts|michigan|minnesota|mississippi|"
        r"missouri|montana|nebraska|nevada|new hampshire|new jersey|new mexico|"
        r"north carolina|north dakota|ohio|oklahoma|oregon|pennsylvania|"
        r"rhode island|south carolina|south dakota|tennessee|texas|utah|vermont|"
        r"virginia|washington|west virginia|wisconsin|wyoming)\b",
        # "City, ST" state abbreviations
        r",\s*(?:AL|AK|AZ|AR|CA|CO|CT|DE|FL|GA|HI|ID|IL|IN|IA|KS|KY|LA|ME|MD|MA|"
        r"MI|MN|MS|MO|MT|NE|NV|NH|NJ|NM|NY|NC|ND|OH|OK|OR|PA|RI|SC|SD|TN|TX|UT|"
        r"VT|VA|WA|WV|WI|WY)\b",
        r"\b[A-Z]{2}\s+\d{5}\b",  # US state + ZIP
    ],
}

_CURRENCY = {
    "India": "INR (₹)", "US": "USD ($)", "UK": "GBP (£)",
    "EU": "EUR (€)", "Canada": "CAD (C$)", "Australia": "AUD (A$)",
    "Global": "the role's local currency",
}


def _detect_market(text: str, location: str = "") -> str:
    """Best-effort detection of the role's country market."""
    blob = f"{location}\n{text[:4000]}"
    scores = {
        market: sum(1 for p in pats if re.search(p, blob, re.IGNORECASE))
        for market, pats in _MARKET_SIGNALS.items()
    }
    best = max(scores.values())
    if best == 0:
        return "Global"
    # Deterministic priority on ties: more-specific currencies/regions before US
    # (US uses $, which is ambiguous with CAD/AUD, so it goes last).
    for market in ("India", "UK", "EU", "Canada", "Australia", "US"):
        if scores[market] == best:
            return market
    return "Global"


def _market_from_url(url: str) -> str:
    """Hint the market from the job board's domain / TLD when JD content is
    ambiguous. Returns a market name or '' if no confident hint."""
    if not url:
        return ""
    try:
        host = (urllib.parse.urlparse(url).netloc or "").lower()
    except Exception:
        return ""
    # Country-code TLDs are strong signals
    if host.endswith((".nl", ".de", ".fr", ".es", ".it", ".be", ".ie", ".at", ".pt")):
        return "EU"
    if host.endswith((".co.uk", ".uk")):
        return "UK"
    if host.endswith((".in",)) or "naukri" in host or "instahyre" in host or "iimjobs" in host:
        return "India"
    if host.endswith((".ca",)):
        return "Canada"
    if host.endswith((".com.au", ".au")):
        return "Australia"
    # US-centric boards default to US (not India)
    if any(b in host for b in ("ziprecruiter", "indeed.com", "dice.com", "monster.com",
                               "greenhouse.io", "lever.co", "glassdoor.com")):
        return "US"
    return ""


def _build_prompt(jd_text: str, signals: dict, onet: dict, wages: dict, demos: str,
                  fallback: bool, market: str = "Global", salary_data: dict = None,
                  income_data: dict = None) -> str:
    parts = [SYSTEM_PROMPT, "\n\n=== INPUT DATA ==="]

    if market == "Global":
        parts.append(
            "\nDETECTED_MARKET: Global (undetermined). The market could NOT be determined "
            "from the JD or URL. Do NOT default to India or any specific country — in "
            "particular do NOT use ₹/INR or name Indian companies. Use USD ($) as a neutral "
            "currency, US-style income tiers, and globally-recognizable employers, and note "
            "in evidence that the location/market was not specified."
        )
    else:
        parts.append(
            f"\nDETECTED_MARKET: {market}. CRITICAL: every currency and income figure in "
            f"your output MUST be expressed in {_CURRENCY.get(market, 'the local currency')}. "
            f"Do NOT use USD unless DETECTED_MARKET is US. Name employers and sourcing "
            f"channels native to this market. Use local household-income tiers and salary "
            f"conventions (e.g. LPA for India, ZZP day-rates for EU/NL)."
        )

    if fallback:
        parts.append(f"\nFALLBACK MODE: URL extraction failed. Inferred title: {signals.get('title', 'Unknown')}. Generate personas from title + industry heuristics.")
    else:
        parts.append(f"\nCLEAN_JD_MARKDOWN:\n{jd_text[:4000]}")

    parts.append(
        "\nSTRUCTURED_JD (extracted directly from the posting — AUTHORITATIVE). "
        "Use these values VERBATIM. Never output \"Not specified\" for a field present here. "
        "A salary or location present here OVERRIDES all inferred or benchmark figures, "
        "and the responsibilities below define the role's true operating center:\n"
        + json.dumps({k: v for k, v in signals.items() if v}, indent=2)
    )

    if onet:
        parts.append(f"\nONET_GROUNDING:\n{json.dumps(onet, indent=2)}")
    if wages:
        salary_str = (
            f"25th percentile: ${wages.get('percentile_25','?')}, "
            f"Median: ${wages.get('median','?')}, "
            f"75th percentile: ${wages.get('percentile_75','?')} "
            f"({wages.get('location','')}"
        )
        parts.append(
            f"\nMARKET_GROUNDING.salary_bounds (SECONDARY benchmark only — a US national "
            f"aggregate; if the JD states its own salary/location, prefer the JD): {salary_str}"
        )
    if salary_data and salary_data.get("mean"):
        rng = ""
        if salary_data.get("low") and salary_data.get("high"):
            rng = f", typical range {salary_data['low']:,}–{salary_data['high']:,}"
        parts.append(
            f"\nADZUNA_SALARY (REAL external benchmark — actually queried for this title × "
            f"location in {salary_data.get('country','')}; treat as SOURCED and HIGH confidence, "
            f"and cite as 'Adzuna market data'): average {salary_data['mean']:,}{rng} "
            f"across {salary_data.get('count','many')} live postings. Use this to ground "
            f"target_monthly_income and household income; you MAY name Adzuna as the source."
        )
    if income_data and income_data.get("us_median_household_income"):
        parts.append(
            f"\nFRED_INCOME (REAL US median household income, source FRED, {income_data.get('year','')}): "
            f"${income_data['us_median_household_income']:,}. Use this as the anchor when assigning "
            f"US household-income tiers; you MAY cite FRED as the source."
        )
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
    "metro_area": "string — the specific city/metro, never a national aggregate",
    "posted_compensation": "string — the salary/comp stated in the JD, verbatim, or null if none stated",
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
  "job_ad_rewrite": {
    "current_jd_risk": "string",
    "missing_motivator": "string",
    "recommended_headline": "string",
    "bullet_to_add": "string",
    "bullet_to_remove": "string",
    "cta_improvement": "string"
  },
  "persona_jd_mismatch": {
    "you_want": "string — the candidate the company is clearly trying to hire, in one phrase",
    "jd_attracts": "string — who the JD as written will ACTUALLY attract (often broader/different)",
    "mismatch": "string — the specific gap between the two (what the JD under- or over-states)",
    "fix": "string — the concrete JD change that closes the gap"
  },
  "personas": [
    {
      "metadata": {
        "name": "string — a clear COHORT/SEGMENT LABEL for this pool (e.g. 'First-Job Crew', 'Experienced QSR Crew', 'Student Part-Timer', 'Displaced Manager'). NEVER a personal/human name (no 'Alex Rivera').",
        "archetype": "string — a short prior-background descriptor for the pool (e.g. 'First job, no experience', 'Switching from a competitor'), NOT a personality type",
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
        "household_income_note": "string — financial pressure/motivation implication, not a restatement of the number",
        "target_monthly_income_from_role": "string — estimated GROSS MONTHLY income from THIS role, as a DOLLAR RANGE that reflects variable shifts/hours, e.g. '$1,800–$2,400'. ALWAYS a range, never a single flat figure, and never a dependency word like 'Primary'.",
        "income_dependency": "Primary|Secondary|Supplemental",
        "hours_per_week_expected": "string — GIG/HOURLY/FRONTLINE ONLY; set null for salaried/corporate",
        "payment_preference": "Daily instant|Weekly|Monthly — GIG/HOURLY/FRONTLINE ONLY; set null for salaried/corporate"
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
      "deal_breakers": ["string — from the CANDIDATE'S perspective: what about THIS role would make THIS specific segment decline or not apply (their walk-away triggers), e.g. a remote-first agency lead: 'mandatory 5-day on-site'; a senior pro: 'no real brand ownership / too junior a title'; a gig worker: 'pay below their current app'. These are the candidate's objections, NOT the employer's requirements, and MUST differ meaningfully across personas. 3-5 items."],
      "candidate_journey": {
        "discovery_risk": "string — why this segment may never SEE the role (wrong channels, weak title)",
        "click_risk": "string — why they may ignore the ad even if they see it",
        "apply_risk": "string — why they may abandon the application form",
        "interview_risk": "string — why they may disengage during interviews",
        "offer_risk": "string — why they may reject the offer",
        "early_churn_risk": "string — why they may leave within the first 90 days"
      },
      "tech_profile": {
        "key_apps": ["string — apps/platforms this segment lives in, useful for sourcing"],
        "tech_savviness_score": "integer 1-5 — GIG/HOURLY/FRONTLINE ONLY; set null for corporate/professional",
        "hardware_devices": "array — GIG/HOURLY/FRONTLINE ONLY (e.g. budget Android); set null for corporate/professional"
      },
      "evidence_confidence": {
        "overall_score": 80,
        "sourced_vs_inferred": "string — plainly state what is grounded in the JD/data vs what is inferred",
        "confidence_breakdown": {
          "high": ["string — fields you are confident about, typically JD-sourced: e.g. 'Location', 'Salary', 'Experience requirement'"],
          "medium": ["string — partially grounded: e.g. 'Source companies', 'Likely motivation'"],
          "low": ["string — mostly inferred: e.g. 'Household income', 'Age range', 'Churn trigger'"]
        },
        "evidence_basis": ["string — ONLY sources actually supplied in INPUT DATA; otherwise exactly 'Inferred from JD requirements; external salary source not used'"],
        "notes": "string"
      },
      "screening_question": {
        "question": "string — recommended diagnostic interview/onboarding question for THIS segment",
        "high_risk_answer": "string — the response pattern that is a RED FLAG (e.g. focuses only on vanity metrics or short-term launch spikes)",
        "risk_rationale": "string — why that answer predicts poor fit or early churn for THIS specific role",
        "why_it_matters": "string — what a strong answer reveals"
      },
      "recruiter_action": {
        "sourcing_channel": {
          "primary": "string — exact platform AND how to use it (e.g. 'LinkedIn Recruiter')",
          "search_string": "string — a copy-paste boolean search: role titles + industry terms + skills (e.g. (\"Brand Manager\" OR \"Campaign Manager\") AND (CPG OR FMCG OR beverage) AND (\"global campaign\" OR \"media planning\"))",
          "target_companies": ["string — 5-8 specific companies/talent pools to source this persona from"],
          "organic_play": "string — non-paid tactic"
        },
        "conversion_hook": {
          "headline": "string — persona-specific ad headline (different for each persona)",
          "core_value_prop": "string — the pitch that resonates with THIS pool specifically"
        },
        "outreach_script": "string — a ready-to-send 3-sentence outreach/InMail tuned to THIS segment's tone and primary motivation (consultative/executive for senior pools; stability+ownership for agency leavers; benefits+steady-pay for frontline). Reference the JD's best perks. No fabricated specifics.",
        "application_dropoff_risk": {
          "risk": "string — why THIS segment abandons before applying or declines the offer",
          "fix": "string — the specific change to JD/process/messaging that removes that drop-off"
        },
        "funnel_friction_killer": "string — exact application process change"
      }
    }
  ]
}'''
    parts.append(schema)
    return "\n".join(parts)


# ── Stage 5: LLM call (Gemini → Groq) ──────────────────────────────────────

def _call_gemini(prompt: str, timeout: int = LLM_TIMEOUT, key: str = None) -> str:
    """Call Gemini Flash via OpenAI-compatible endpoint."""
    resp = requests.post(
        "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
        headers={"Authorization": f"Bearer {key or GEMINI_KEY}", "Content-Type": "application/json"},
        json={
            "model": GEMINI_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": LLM_MAX_TOKENS,
            "response_format": {"type": "json_object"},
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def _call_groq(prompt: str, timeout: int = LLM_TIMEOUT) -> str:
    """Call Groq Llama 3.3 70B."""
    resp = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {GROQ_KEY}", "Content-Type": "application/json"},
        json={
            "model": "llama-3.3-70b-versatile",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": LLM_MAX_TOKENS,
            "response_format": {"type": "json_object"},
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def _call_openai_compatible(url: str, key: str, model: str, prompt: str, extra_headers: dict = None,
                            timeout: int = LLM_TIMEOUT) -> str:
    """Generic caller for any OpenAI-compatible /chat/completions endpoint."""
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    if extra_headers:
        headers.update(extra_headers)
    resp = requests.post(
        url,
        headers=headers,
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": LLM_MAX_TOKENS,
            "response_format": {"type": "json_object"},
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def _call_cerebras(prompt: str, timeout: int = LLM_TIMEOUT) -> str:
    """Cerebras — fast Llama 3.3 70B, generous free tier (1M tokens/day)."""
    return _call_openai_compatible(
        "https://api.cerebras.ai/v1/chat/completions",
        CEREBRAS_KEY,
        os.environ.get("CEREBRAS_MODEL", "gpt-oss-120b"),
        prompt,
        timeout=timeout,
    )


def _call_openrouter(prompt: str, timeout: int = LLM_TIMEOUT) -> str:
    """OpenRouter — aggregator with free models."""
    return _call_openai_compatible(
        "https://openrouter.ai/api/v1/chat/completions",
        OPENROUTER_KEY,
        os.environ.get("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct:free"),
        prompt,
        extra_headers={
            "HTTP-Referer": "https://nova-persona.vercel.app",
            "X-Title": "Nova Candidate Map",
        },
        timeout=timeout,
    )


def _call_github(prompt: str, timeout: int = LLM_TIMEOUT) -> str:
    """GitHub Models — free with a GitHub token."""
    return _call_openai_compatible(
        os.environ.get("GITHUB_MODELS_URL", "https://models.github.ai/inference/chat/completions"),
        GITHUB_MODELS_KEY,
        os.environ.get("GITHUB_MODELS_MODEL", "openai/gpt-4o-mini"),
        prompt,
        timeout=timeout,
    )


def _call_anthropic(prompt: str, timeout: int = LLM_TIMEOUT) -> str:
    """Anthropic Claude via the Messages API (not OpenAI-compatible)."""
    resp = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={"x-api-key": ANTHROPIC_KEY, "anthropic-version": "2023-06-01", "Content-Type": "application/json"},
        json={
            "model": os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001"),
            "max_tokens": LLM_MAX_TOKENS, "temperature": 0.7,
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()["content"][0]["text"]


# Additional OpenAI-compatible providers (name, key, url, model-env, default-model).
# Trimmed to providers that actually work or are fixable. Removed dead weight that
# consistently failed: mistral/nvidia/zhipu (chronic timeouts), together/sambanova/
# siliconflow (no account credit → 402/403). Re-add them only with funded keys.
# Model strings updated to current valid ones (the old defaults had been retired:
# grok-2-latest → grok-4.3; command-r-plus → command-a-03-2025).
_EXTRA_OAI = [
    # DeepSeek first among the paid providers — cheap + reliable, so it's the
    # preferred paid fallback BEFORE the low-balance Anthropic floor.
    ("deepseek", DEEPSEEK_KEY, "https://api.deepseek.com/v1/chat/completions",          "DEEPSEEK_MODEL", "deepseek-v4-flash"),
    ("openai",   OPENAI_KEY,   "https://api.openai.com/v1/chat/completions",              "OPENAI_MODEL",   "gpt-4o-mini"),
    ("xai",      XAI_KEY,      "https://api.x.ai/v1/chat/completions",                    "XAI_MODEL",      "grok-4.3"),
    # cohere removed — repeatedly 404'd then timed out, wasting ~18s right before the
    # Anthropic floor. Re-add only if it proves reliable.
]


def _make_oai_caller(url, key, model):
    return lambda prompt, timeout=LLM_TIMEOUT: _call_openai_compatible(url, key, model, prompt, timeout=timeout)


def _call_llm(prompt: str) -> str:
    """Multi-provider fallback chain, with graceful handling of upstream rate limits.

    Tries each configured provider; on a 429 it retries once, honoring the
    provider's Retry-After header (capped at 5s). If every provider is
    rate-limited or unavailable, raises LLMUnavailableError (→ HTTP 503) so the
    user sees a clear "try again shortly" message instead of a generic 500.
    """
    # Fallback order: most generous/reliable free tiers first.
    # Order = quality-first, with generous-free fallbacks behind it. Gemini leads
    # because it produces stronger sociological segmentation than gpt-oss-120b.
    providers = []
    # One entry per Gemini key (each from a separate Google project = its own free
    # quota). A 429 on one key falls through to the next key, then the next provider.
    for _i, _gk in enumerate(GEMINI_KEYS):
        _label = "gemini" if _i == 0 else f"gemini{_i + 1}"
        providers.append((_label, lambda p, t, _k=_gk: _call_gemini(p, t, key=_k)))
    if CEREBRAS_KEY:
        providers.append(("cerebras", _call_cerebras))
    if GROQ_KEY:
        providers.append(("groq", _call_groq))
    if OPENROUTER_KEY:
        providers.append(("openrouter", _call_openrouter))
    if GITHUB_MODELS_KEY:
        providers.append(("github", _call_github))
    for _name, _key, _url, _menv, _mdef in _EXTRA_OAI:
        if _key:
            providers.append((_name, _make_oai_caller(_url, _key, os.environ.get(_menv, _mdef))))
    # Anthropic is a PAID, low-balance key — keep it as the ABSOLUTE LAST resort so
    # it only spends credit when every free and other provider has failed.
    if ANTHROPIC_KEY:
        providers.append(("anthropic", _call_anthropic))

    if not providers:
        raise RuntimeError(
            "No LLM key configured. Set one of CEREBRAS_API_KEY, GEMINI_API_KEY, "
            "GROQ_API_KEY, OPENROUTER_API_KEY, or GITHUB_MODELS_TOKEN."
        )

    errors = []
    _last_idx = len(providers) - 1
    for _idx, (name, fn) in enumerate(providers):
        # Fail FAST on every provider except the final one — a hung early provider
        # must not consume the budget. The FINAL last-resort provider (usually Claude)
        # gets the big remaining window so it can actually finish a full map. Every
        # call is still capped by the time left, so we always return our own JSON.
        remaining = (_REQUEST_DEADLINE - time.time()) if _REQUEST_DEADLINE else float(LLM_TIMEOUT)
        cap = LLM_TIMEOUT if _idx == _last_idx else LLM_TIMEOUT_FAST
        call_timeout = int(min(cap, remaining - 4))
        if call_timeout < 5:
            errors.append((name, requests.exceptions.Timeout("request budget exhausted")))
            print(f"[nova] stopping provider chain before {name}: request budget nearly exhausted")
            break
        for attempt in (1, 2):
            try:
                return fn(prompt, call_timeout)
            except requests.exceptions.HTTPError as e:
                code = getattr(e.response, "status_code", None)
                if code == 429 and attempt == 1:
                    wait = 2.0
                    ra = e.response.headers.get("Retry-After") if e.response is not None else None
                    if ra:
                        try:
                            wait = min(float(ra), 5.0)
                        except ValueError:
                            pass
                    print(f"[nova] {name} rate-limited (429), retrying in {wait}s")
                    time.sleep(wait)
                    continue
                errors.append((name, e))
                print(f"[nova] {name} failed ({e})")
                break
            except Exception as e:
                errors.append((name, e))
                print(f"[nova] {name} failed ({e})")
                break

    # Every configured provider failed. Report the TRUE cause — this is an internal
    # tool, so an honest "all providers exhausted / timed out" beats a vague message
    # or (worse) fabricating a result.
    def _reason(e):
        if isinstance(e, requests.exceptions.HTTPError):
            sc = getattr(e.response, "status_code", None)
            if sc == 429:
                return "rate-limited (429)"
            if sc in (401, 402, 403):
                return f"quota/auth ({sc})"
            return f"HTTP {sc}"
        if isinstance(e, requests.exceptions.Timeout):
            return "timed out"
        if isinstance(e, requests.exceptions.ConnectionError):
            return "connection error"
        return type(e).__name__

    n = len(errors)
    n_rl = sum(1 for _, e in errors if isinstance(e, requests.exceptions.HTTPError)
               and getattr(e.response, "status_code", None) == 429)
    n_to = sum(1 for _, e in errors if isinstance(e, requests.exceptions.Timeout))
    summary = "; ".join(f"{name}: {_reason(e)}" for name, e in errors) or "no providers configured"

    if n and n_rl == n:
        raise LLMUnavailableError(
            f"AI generation failed: all {n} model providers are rate-limited / quota-exhausted "
            f"right now. Please wait a minute and retry. [{summary}]"
        )
    if n and n_to == n:
        raise LLMUnavailableError(
            f"AI generation failed: all {n} model providers timed out (no response within the time "
            f"limit). Please retry shortly. [{summary}]"
        )
    if n and (n_rl + n_to) == n:
        raise LLMUnavailableError(
            f"AI generation failed: all {n} model providers were exhausted (rate-limited) or timed "
            f"out. Please retry shortly. [{summary}]"
        )
    if any(isinstance(e, (requests.exceptions.Timeout, requests.exceptions.ConnectionError,
                          requests.exceptions.HTTPError)) for _, e in errors):
        raise LLMUnavailableError(
            f"AI generation failed: every model provider errored. Please try again. [{summary}]"
        )
    # Unknown failure — re-raise so it surfaces (and gets logged) as a 500.
    raise errors[-1][1]


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
    """Extract and parse JSON from an LLM response, repairing common malformations
    (missing/trailing commas, truncated tails) that weaker models emit."""
    # 1) Direct parse
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    # 2) Largest {...} block
    m = re.search(r"\{[\s\S]*\}", raw)
    candidate = m.group() if m else raw
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass
    # 3) Repair malformed LLM JSON (missing commas, trailing commas, etc.)
    try:
        from json_repair import repair_json
        obj = repair_json(candidate, return_objects=True)
        if isinstance(obj, dict) and obj:
            return obj
    except Exception:
        pass
    # Clean, user-facing failure instead of a 500 stack trace.
    raise OutputValidationError(
        "The AI model returned a malformed response. Please try again."
    )


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

    # Only `personas` is truly essential. Backfill the rest so a truncated tail
    # (e.g. a missing job_ad_rewrite on a large 6-persona map) degrades
    # gracefully — the frontend renders these conditionally — instead of 502-ing.
    data.setdefault("jd_hard_filters", {})
    data.setdefault("role_summary", "")
    data.setdefault("recruiter_brief", "")
    data.setdefault("local_context", {})
    data.setdefault("diversity_scoring", {})
    data.setdefault("job_ad_rewrite", {})
    data.setdefault("persona_jd_mismatch", {})

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


def _target_persona_count(preset: str, score) -> int:
    """Preset-aware deterministic persona count from the diversity score.
    Frontline/gig roles run wider (5-6); corporate/exec run tighter (3-4/3)."""
    try:
        score = int(score)
    except (TypeError, ValueError):
        score = 8
    p = (preset or "").lower()
    if "executive" in p:
        return 3
    if "gig" in p or "hourly" in p or "frontline" in p:
        return 6 if score >= 10 else 5
    if "licensed" in p:
        return 3 if score <= 7 else (4 if score <= 11 else 5)
    # corporate_professional and default
    return 3 if score <= 7 else 4


def _enforce_persona_count(data: dict) -> dict:
    """Deterministically cap the number of personas to the preset+score target.
    Trims surplus personas (smallest segments first, keeping any bridge) and
    re-normalizes the segment percentages. Never fabricates personas."""
    ds = data.get("diversity_scoring") or {}
    personas = data.get("personas")
    if not isinstance(personas, list) or not personas:
        return data
    target = _target_persona_count(ds.get("preset_used", ""), ds.get("total_score", 8))
    if len(personas) > target:
        def _seg(p):
            return (p.get("metadata") or {}).get("segment_size_percentage", 0) or 0
        def _is_bridge(p):
            return bool((p.get("metadata") or {}).get("is_bridge_persona"))
        # Keep bridge personas + the largest segments up to target.
        ordered = sorted(personas, key=lambda p: (_is_bridge(p), _seg(p)), reverse=True)
        personas = ordered[:target]
        data["personas"] = personas
        data = _normalize_segments(data)
    if isinstance(ds, dict):
        ds["target_persona_count"] = len(personas)
    return data


_INTERNAL_TOKENS = re.compile(
    r"(MARKET[_ ]?GROUNDING(?:[._ ]?demographic[_ ]?signals)?"
    r"|STRUCTURED[_ ]?JD|ONET[_ ]?GROUNDING)",
    re.IGNORECASE,
)

# Salary-data sources the model fabricates (we never query these). NOTE: bare
# job-board names (Indeed, Glassdoor, ZipRecruiter, LinkedIn) are LEGITIMATE
# sourcing channels, so they're only scrubbed when paired with "salary".
_FAKE_SOURCES = re.compile(
    r"\b(Salary ?Expert|Pay ?scale|Salary\.com|Levels\.fyi|Comparably"
    r"|(?:Glassdoor|Indeed|ZipRecruiter|LinkedIn)\s+salary(?:\s+data)?"
    r"|Bureau of Labor Statistics|BLS(?:\s+data)?)\b[^,;|]*",
    re.IGNORECASE,
)


def _scrub_internal_tokens(value):
    """Strip internal prompt variable names AND fabricated source names that a
    model may echo into output strings, so neither leaks into the UI."""
    if isinstance(value, str):
        s = _INTERNAL_TOKENS.sub("US labor-market norms", value)
        s = _FAKE_SOURCES.sub("US labor-market norms", s)
        # Remove leaked placeholders like "(truncated text)" / "(incomplete requirement)".
        s = re.sub(r"\s*\((?:truncated|incomplete)[^)]*\)", "", s, flags=re.IGNORECASE)
        return s.strip()
    if isinstance(value, dict):
        return {k: _scrub_internal_tokens(v) for k, v in value.items()}
    if isinstance(value, list):
        cleaned = [_scrub_internal_tokens(v) for v in value]
        # drop emptied entries, then de-dupe consecutive identical strings
        out = []
        for x in cleaned:
            if isinstance(x, str) and not x.strip():
                continue
            if out and isinstance(x, str) and x == out[-1]:
                continue
            out.append(x)
        return out
    return value


# ── Main entry point ────────────────────────────────────────────────────────

def _parse_money_range(s: str):
    """Parse a household-income range string ('$80,000–$110,000', '$80k-$110k')
    into (low, high) integer dollars, or None."""
    vals = []
    for num, k in re.findall(r"\$?\s*([\d][\d,.]*)\s*([kK])?", s or ""):
        n = num.replace(",", "")
        if not n or n == ".":
            continue
        try:
            v = float(n)
        except ValueError:
            continue
        if k or v < 1000:        # "80k" or a bare "80" → 80,000
            v *= 1000
        vals.append(int(round(v)))
    vals = [v for v in vals if v >= 1000]
    return (min(vals), max(vals)) if len(vals) >= 2 else None


def _drop_truncated_personas(data: dict) -> dict:
    """Safety net: if the model output was truncated mid-persona (e.g. a provider's
    output-token ceiling), the trailing persona has demographics/financials but no
    later fields. Drop any persona missing BOTH its drivers/motivations AND evidence
    blocks rather than render a half-empty card. Keeps at least one persona."""
    personas = data.get("personas")
    if not isinstance(personas, list) or len(personas) <= 1:
        return data

    def _complete(p):
        if not isinstance(p, dict):
            return False
        return bool(p.get("drivers_and_friction")) or bool(p.get("evidence_confidence"))

    kept = [p for p in personas if _complete(p)]
    if kept and len(kept) < len(personas):
        data["personas"] = kept
    return data


_BRIDGE_KW = re.compile(r"bridge|displaced|overqualified|laid.?off|former\s+(?:manager|leader)|stop.?gap", re.I)

def _fix_persona_display(data: dict) -> dict:
    """Deterministic display guards for two model slips:
    (1) a stray is_bridge_persona flag on a CORE pool (e.g. 'First-Job Crew' tagged
        Bridge) — keep the bridge tag only on a persona whose name/archetype actually
        reads as an overqualified/displaced bridge;
    (2) the 'Expected Monthly' figure (target_monthly_income_from_role) populated with
        a dependency word like 'Primary'/'Supplemental' instead of a dollar value —
        blank it so the card never shows a non-money value in a money field."""
    for p in data.get("personas", []):
        if not isinstance(p, dict):
            continue
        meta = p.get("metadata") or {}
        fin = p.get("financials") if isinstance(p.get("financials"), dict) else {}
        # (1) strip a bogus bridge tag from a non-bridge persona
        if meta.get("is_bridge_persona"):
            label = f"{meta.get('name','')} {meta.get('archetype','')}"
            if not _BRIDGE_KW.search(label):
                meta["is_bridge_persona"] = False
        # (2) Expected Monthly must be a DOLLAR RANGE. Blank a non-money value (e.g.
        #     'Primary'); expand a single dollar figure into a ±15% range.
        tm = fin.get("target_monthly_income_from_role")
        if isinstance(tm, str):
            nums = [int(x.replace(",", "")) for x in re.findall(r"[\d,]{2,}", tm)
                    if x.replace(",", "").isdigit()]
            nums = [n for n in nums if n >= 100]
            if not nums:
                fin["target_monthly_income_from_role"] = ""
            elif len(nums) == 1:
                n = nums[0]
                lo = int(round(n * 0.85 / 50)) * 50
                hi = int(round(n * 1.15 / 50)) * 50
                fin["target_monthly_income_from_role"] = f"${lo:,}–${hi:,}"
    return data


def _cap_hourly_household_income(data: dict) -> dict:
    """DETERMINISTIC backstop: for hourly/gig roles the LLM keeps inflating
    household income for income-dependent personas. Cap any over-high range down
    (preserving its width) so a working-parent/crew persona can't sit at six
    figures and the bridge can't exceed a modest ceiling. Lowers only; never raises."""
    lc = data.get("local_context") or {}
    if (lc.get("role_type") or "").lower() not in ("hourly", "gig"):
        return data
    for p in data.get("personas", []):
        fin = p.get("financials")
        if not isinstance(fin, dict):
            continue
        rng = fin.get("household_income_range")
        parsed = _parse_money_range(rng) if isinstance(rng, str) else None
        if not parsed:
            continue
        low, high = parsed
        is_bridge = bool((p.get("metadata") or {}).get("is_bridge_persona"))
        ceiling = 105000 if is_bridge else 75000
        if high <= ceiling:
            continue
        width = max(high - low, 10000)
        new_high = ceiling
        new_low = max(new_high - width, 15000)
        use_k = "k" in rng.lower()
        if use_k:
            fin["household_income_range"] = f"${new_low // 1000}k–${new_high // 1000}k"
        else:
            fin["household_income_range"] = f"${new_low:,}–${new_high:,}"
    return data


def build_persona_response(text: str = "", url: str = "", source: str = "job_description",
                           mode: str = "job_description") -> dict:
    """
    Main pipeline orchestrator.
    Args:
        text: raw job description text (if pasted)
        url:  job posting URL (if provided)
    Returns:
        Full persona payload dict matching PRD v2.1 schema
    """
    start = time.time()
    globals()["_REQUEST_DEADLINE"] = start + REQUEST_BUDGET_S
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

        # Input classification: a search/category results page is NOT a single JD.
        # Instead of hard-failing, offer the user a choice (market map / pick a job /
        # paste JD). If they already chose "market_map", synthesize a cluster brief.
        if _looks_like_search_page(url, jd_content):
            cluster = _role_cluster_from_url(url)
            if mode != "market_map":
                mkt = _detect_market(jd_content, "")
                if mkt == "Global":
                    mkt = _market_from_url(url) or "Global"
                raise SearchPageError(role_hint=cluster, market=mkt)
            cluster = cluster or "this role category"
            jd_content = (
                f"GENERAL MARKET MAP REQUEST (no single job description provided). Produce a "
                f"broad, realistic candidate-market map for '{cluster}' roles: the full spectrum "
                f"of candidate pools for this category, each with a sourcing playbook and "
                f"conversion guidance. Treat '{cluster}' as the role focus."
            )
            fallback = False
            inferred_title = cluster
            result["_pipeline"]["mode"] = "market_map"

        # For a user-provided URL, only reject fetched content on SECURITY grounds
        # (prompt injection or junk/garbage). We deliberately do NOT apply the
        # stricter "looks like a job description" gate here — the user explicitly
        # asked us to analyze this page, and the content is sanitized before it
        # ever reaches the LLM. If the content is unsafe, fall back to the title
        # inferred from the URL slug instead of failing.
        if jd_content and not fallback and (_check_injection(jd_content) or _is_repetitive_garbage(jd_content)):
            fallback = True
            inferred_title = inferred_title or _extract_title_from_url(url)
            jd_content = inferred_title
            result["_pipeline"]["fallback_reason"] = "fetched_page_unsafe"

    if not jd_content and not inferred_title:
        # Nothing usable came back from the URL and no title could be inferred.
        raise SafetyError(
            "We couldn't read anything from that URL. Please paste the job "
            "description text directly instead."
        )

    # ── Sanitize before injecting into LLM prompt ──────────────────────────
    jd_content = sanitize_for_prompt(jd_content)

    # Stage 2: Extract signals
    signals = _extract_signals(jd_content)
    # Use the page/slug title whenever the in-text title regex came up empty
    # (e.g. Greenhouse pages where the body starts with legal boilerplate).
    if inferred_title:
        # The page/ATS title is more reliable than the in-text regex guess.
        signals["title"] = inferred_title
    result["_pipeline"]["signals"] = signals

    # Hard guard: in fallback mode we rely entirely on the title, so never
    # generate personas from a junk/numeric title (e.g. a Greenhouse job ID).
    if fallback and _is_invalid_title(signals.get("title", "")):
        raise SafetyError(
            "We couldn't extract the job description from that URL. "
            "Please paste the full job description text instead."
        )

    # Stage 3: O*NET grounding
    onet = _onet_grounding(signals.get("title", ""))
    result["_pipeline"]["onet_soc"] = onet.get("soc_code", "")

    # Detect the role's market so we use the right currency and wage source
    market = _detect_market(jd_content, signals.get("location", ""))
    if market == "Global":
        # Fall back to the job board's domain/TLD before giving up on geo.
        market = _market_from_url(url) or "Global"
    result["_pipeline"]["market"] = market

    # Stage 4: CareerOneStop wages — US-only (USD) source. Only use it for US
    # roles; otherwise it would anchor the model to USD figures for non-US markets.
    wages = _careeronestop_wages(signals.get("title", ""), signals.get("location", "")) if market == "US" else {}
    result["_pipeline"]["wages"] = wages

    # Stage 4b: real grounding (only if enrichment enabled) — Adzuna salary (global) + FRED income (US)
    salary_data = _enrich_salary(signals.get("title", ""), signals.get("location", ""), market, jd_content)
    result["_pipeline"]["adzuna"] = salary_data
    _adzuna_diag = dict(_LAST_ADZUNA_DIAG)  # TEMP DEBUG snapshot
    income_data = _enrich_income(market)
    result["_pipeline"]["fred"] = income_data

    # Stage 5: Tavily demographics
    demos = _tavily_demographics(signals.get("title", ""), signals.get("location", ""))
    result["_pipeline"]["tavily_chars"] = len(demos)

    # Stage 6: Build prompt and call LLM
    prompt = _build_prompt(jd_content, signals, onet, wages, demos, fallback, market, salary_data, income_data)
    raw = _call_llm(prompt)

    # Stage 7: Parse, validate, and normalize
    data = _parse_json(raw)
    data = validate_output(data)      # schema + safety check before it leaves the server
    data = _drop_truncated_personas(data)  # never render a half-generated trailing persona
    data = _fix_persona_display(data)      # strip stray bridge tags; clean non-money 'Expected Monthly'
    data = _normalize_segments(data)  # renormalizes % after any drop
    data = _enforce_persona_count(data)   # deterministic, preset-aware count cap
    data = _scrub_internal_tokens(data)   # strip any leaked internal variable names

    # Guarantee the posted salary + location show even if the model omitted them.
    lc = data.get("local_context")
    if isinstance(lc, dict):
        if not lc.get("posted_compensation") and signals.get("salary"):
            lc["posted_compensation"] = signals["salary"]
        if not lc.get("metro_area") and signals.get("location"):
            lc["metro_area"] = signals["location"]

    # Deterministic income backstop — the prompt can't reliably hold hourly HH down,
    # so cap it here (working parent / income-dependent ≤ $75k, bridge ≤ $105k).
    data = _cap_hourly_household_income(data)

    # Resolve the best title + location we have, preferring deterministic signals
    # but falling back to the LLM's own role_summary / metro when extraction missed.
    _lcd = lc if isinstance(lc, dict) else {}
    resolved_title = (signals.get("title") or _title_from_summary(data.get("role_summary", ""))).strip()
    resolved_loc = (signals.get("location") or _lcd.get("metro_area") or "").strip()

    # Post-LLM enrichment RETRY: the pre-LLM pass needs title + a country market,
    # which deterministic extraction sometimes misses (no labeled title line, or a
    # metro the signal list didn't know). Now that we have the LLM's title + metro,
    # re-detect the market and try Adzuna once more before giving up.
    #
    # IMPORTANT: only retry when the pre-LLM pass NEVER actually queried Adzuna
    # (it was gated off for a missing title or undetermined country). If it already
    # ran the city→state→national chain and came back empty, re-running it would
    # just repeat the same calls and risk the function's time budget — so skip.
    _pre_reason = (_adzuna_diag or {}).get("reason", "")
    _pre_gated = _pre_reason.startswith(("gated_off", "no_adzuna_country"))
    if (ENABLE_ENRICHMENT and not (salary_data and salary_data.get("mean"))
            and resolved_title and _pre_gated):
        market2 = market
        if market2 == "Global":
            market2 = _detect_market(f"{data.get('role_summary','')}\n{resolved_loc}", resolved_loc)
            if market2 == "Global":
                market2 = _market_from_url(url) or "Global"
        if market2 != "Global":
            retry = _enrich_salary(resolved_title, resolved_loc, market2, jd_content)
            result["_pipeline"]["adzuna_retry"] = retry
            _adzuna_diag = dict(_LAST_ADZUNA_DIAG)  # TEMP DEBUG: reflect the retry
            if retry and retry.get("mean"):
                salary_data = retry
                market = market2
                if not income_data:
                    income_data = _enrich_income(market2)

    # Role-level MARKET SALARY RANGE — typical market value for this title × city,
    # taken DETERMINISTICALLY from the Adzuna response (not the LLM) so it is
    # reliable and always carries the real source label. Only attached when the
    # Adzuna call actually returned numbers; otherwise the field is absent and the
    # UI shows nothing (we never fabricate a market range).
    _ms_count = (salary_data or {}).get("count") or 0
    # Plausibility guard: for an explicitly HOURLY / entry-frontline role, a generic
    # title still pulls higher-paid same-word jobs into Adzuna's average. If the
    # annual figure is implausibly high for hourly work, suppress it rather than
    # show a misleading number (better no figure than a wrong one).
    _role_type = ((lc.get("role_type") if isinstance(lc, dict) else "") or "").lower()
    _HOURLY_ANNUAL_CEILING = int(os.environ.get("HOURLY_SALARY_CEILING", "48000"))
    if (salary_data and salary_data.get("mean") and _role_type in ("hourly", "gig")
            and salary_data["mean"] > _HOURLY_ANNUAL_CEILING):
        salary_data = {}  # drop the implausible benchmark for this hourly role
    if salary_data and salary_data.get("mean") and _ms_count >= 1:
        # Confidence scales with how many live postings backed the benchmark.
        if _ms_count >= 50:
            _ms_conf = "High confidence"
        elif _ms_count >= 15:
            _ms_conf = "Medium confidence"
        else:
            _ms_conf = "Directional market signal"
        _CUR = {"US": ("$", "USD"), "GB": ("£", "GBP"), "IN": ("₹", "INR"),
                "CA": ("C$", "CAD"), "AU": ("A$", "AUD"), "NL": ("€", "EUR"),
                "DE": ("€", "EUR"), "FR": ("€", "EUR"), "ES": ("€", "EUR"),
                "IT": ("€", "EUR"), "AT": ("€", "EUR"), "BE": ("€", "EUR"),
                "PL": ("zł", "PLN")}
        _CC_NAME = {"US": "United States", "GB": "United Kingdom", "IN": "India",
                    "CA": "Canada", "AU": "Australia", "NL": "Netherlands",
                    "DE": "Germany", "FR": "France", "ES": "Spain", "IT": "Italy",
                    "AT": "Austria", "BE": "Belgium", "PL": "Poland"}
        cc = (salary_data.get("country") or "").upper()
        sym, code = _CUR.get(cc, ("", ""))
        geo = salary_data.get("geo")
        title_lbl = (resolved_title or "this role").strip()
        loc_lbl = (resolved_loc or "this market").strip()
        if geo == "national":
            geo_lbl = f"{_CC_NAME.get(cc, cc)} (national)"
            note = ("National market average for this title — no city- or state-level salary data "
                    "was available. Not the salary in this posting (which did not state one).")
        elif geo == "state":
            geo_lbl = f"{salary_data.get('geo_name') or 'state'} (statewide)"
            note = ("Statewide market average for this title — no city-level salary data was "
                    "available for this location. Not the salary in this posting (which did not state one).")
        else:
            geo_lbl = salary_data.get("geo_name") or loc_lbl
            note = ("Typical market value for this title in this city, averaged across live job "
                    "postings. Not the salary in this posting (which did not state one).")
        data["market_salary"] = {
            "average": salary_data["mean"],
            "low": salary_data.get("low"),
            "high": salary_data.get("high"),
            "currency_symbol": sym,
            "currency_code": code,
            "country": cc,
            "title": title_lbl,
            "location": geo_lbl,
            "basis": f"{title_lbl} × {geo_lbl}",
            "posting_count": salary_data.get("count"),
            "source": "Adzuna market data",
            "confidence": _ms_conf,
            "scope": geo,
            "note": note,
        }

    # Keep the displayed persona count consistent with the actual cards (no
    # "score says 4 but only 3 cards" contradiction).
    ds = data.get("diversity_scoring")
    personas = data.get("personas")
    if isinstance(ds, dict) and isinstance(personas, list) and personas:
        ds["target_persona_count"] = len(personas)

    # Debug internals — only exposed when DEBUG_PIPELINE is enabled.
    if DEBUG_PIPELINE:
        data["_provider"] = "gemini" if GEMINI_KEY else "groq"
        data["_pipeline_ms"] = round((time.time() - start) * 1000)
        data["_pipeline"] = result["_pipeline"]
        data["_adzuna_diag"] = _adzuna_diag  # inspectable only in debug mode
        return data

    # Production: strip every internal/debug field so nothing leaks to the browser.
    return {k: v for k, v in data.items() if not k.startswith("_")}
