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
GROQ_KEY        = os.environ.get("GROQ_API_KEY", "")
CEREBRAS_KEY    = os.environ.get("CEREBRAS_API_KEY", "")
OPENROUTER_KEY  = os.environ.get("OPENROUTER_API_KEY", "")
# GitHub Models accepts a GitHub PAT; allow either env name.
GITHUB_MODELS_KEY = os.environ.get("GITHUB_MODELS_TOKEN") or os.environ.get("GITHUB_TOKEN", "")

# LLM output ceiling. Default 8000 fits gemini-2.0-flash's 8192 cap. To allow
# fuller 6-persona maps, set GEMINI_MODEL=gemini-2.5-flash and LLM_MAX_TOKENS=16000.
GEMINI_MODEL   = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
LLM_MAX_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", "8000"))

TIMEOUT = 20


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
        sal_m = re.search(r"[\$£€₹]\s?[\d.,]+(?:\s*[-–]\s*[\$£€₹]?\s?[\d.,]+)?(?:\s*[KkMm])?", text)
        if sal_m:
            signals["salary"] = sal_m.group(0).strip()
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
- NO NAMED SALARY SOURCES: never attribute an inferred salary/income figure to a specific website, report, or dataset you were not given (e.g. "SalaryExpert", "Payscale", "Glassdoor", "BLS", "ZipRecruiter salary data"). If a figure is inferred, write exactly "inferred from US labor-market norms" — naming a source you did not receive is fabrication.
- Personas MUST be labor-market segments (real candidate pools with distinct backgrounds and paths to this role), NOT personality archetypes ("The Results Driver", "The Relationship Builder")
- Each persona's archetype AND name must name a SOURCEABLE prior-background talent pool — a real group you could filter for on LinkedIn/Naukri by their last role, industry, or employer type (e.g. "Agency Campaign Operator", "CPG Brand Marketer", "Media-Agency Social Lead", "The Ad Ops Operator", "The SaaS CS Migrant")
- BANNED as an archetype or name: functional skills or personality traits that EVERY candidate for this role would share — e.g. "Data-Driven", "Creative", "Storyteller", "Data-Driven Brand Builder", "Creative Storyteller", "Strategic Thinker", "Results-Driven", "Innovative". These are not distinct candidate pools and cannot be sourced. Litmus test: if you cannot rephrase the archetype as "their last role was X" or "they come from Y industry", it is INVALID — fix it before returning.
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

Sum scores (5–15) and set the persona count strictly from the score within the preset band:
- 3–4 band (corporate_professional): score ≤7 → 3 personas; score ≥8 → 4 personas.
- 3–5 band (licensed_skilled): score ≤7 → 3; 8–11 → 4; ≥12 → 5.
- 5–6 band (gig/hourly): score ≤9 → 5; ≥10 → 6.
- 3 band (executive_specialist): always 3.
GENERATE EXACTLY this many persona objects, and set target_persona_count to the same number. The personas array length MUST equal target_persona_count.

CROSS-AXIS VALIDATION: If Axis B = 3, Axis A CANNOT = 1.

STEP 3 — BRIDGE PERSONA EVALUATION
Include a Bridge Persona if AT LEAST 2 of these 4 signals are true:
1. Low barrier to entry (no degree/license, onboarding in days)
2. Flexible/short-term structure (contract, seasonal, part-time, gig)
3. Economic vulnerability (wages at or below local median, or hourly pay)
4. Broad applicant pool (accepts career changers, no niche experience required)
If triggered: the bridge persona OCCUPIES ONE of the score-dictated persona slots — it does NOT change the total count. Still generate EXACTLY the score-dictated number of personas; one of them is simply the bridge. The bridge should be a MINORITY of the pool (roughly 15–25% segment_size_percentage), not co-equal with the core pools.

CORPORATE / SENIOR BRIDGE (for corporate_professional and executive_specialist presets, where the 4 signals above rarely apply): include a bridge persona when a displaced senior-talent pool is plausible — e.g. an experienced director or manager from the same domain recently affected by layoffs or restructuring who would take this role as an immediate landing spot. They bring strong capability but carry elevated flight risk. Make that risk explicit in their anti_pattern_signals.churn_trigger (e.g. "leaves the moment a director-level seat opens elsewhere"). This bridge is ONE of the score-dictated personas (it does not reduce the count) and should be ~15–25% of the pool.

STEP 4 — GENERATE PERSONAS WITH MAXIMUM VARIANCE
Maximize variance across personas on the highest-scoring axes. Personas that are minor demographic variations of each other are INVALID. Each must represent a genuinely distinct segment with different motivations, financial context, and life stage.

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
✗ REJECT if any archetype/name is a functional skill or trait ("Data-Driven", "Creative Storyteller", "Strategic", "Brand Builder") instead of a sourceable prior-background pool
✗ REJECT if STRUCTURED_JD provides a work_location or salary but you output "Not specified" / omit it — you MUST use the structured value verbatim
✗ REJECT if pew_household_income_tier contradicts household_income_range per the tier bands (e.g. labeling ~$80k "Lower-middle" when that band is $35k–$65k)
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
        r"\b(?:new york|san francisco|seattle|austin|boston|chicago|los angeles)\b",
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
                  fallback: bool, market: str = "Global") -> str:
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
        "household_income_note": "string — financial pressure/motivation implication, not a restatement of the number",
        "target_monthly_income_from_role": "string",
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
      "deal_breakers": ["string — concrete hard no's for THIS segment that would stop them applying or accepting (e.g. 'pay below market', 'no remote flexibility', 'unclear shift schedule', 'no company vehicle', 'too much travel', 'no promotion path'). 3-5 items, specific to this pool and the JD."],
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

def _call_gemini(prompt: str) -> str:
    """Call Gemini Flash via OpenAI-compatible endpoint."""
    resp = requests.post(
        "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
        headers={"Authorization": f"Bearer {GEMINI_KEY}", "Content-Type": "application/json"},
        json={
            "model": GEMINI_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": LLM_MAX_TOKENS,
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
            "max_tokens": LLM_MAX_TOKENS,
            "response_format": {"type": "json_object"},
        },
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def _call_openai_compatible(url: str, key: str, model: str, prompt: str, extra_headers: dict = None) -> str:
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
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def _call_cerebras(prompt: str) -> str:
    """Cerebras — fast Llama 3.3 70B, generous free tier (1M tokens/day)."""
    return _call_openai_compatible(
        "https://api.cerebras.ai/v1/chat/completions",
        CEREBRAS_KEY,
        os.environ.get("CEREBRAS_MODEL", "gpt-oss-120b"),
        prompt,
    )


def _call_openrouter(prompt: str) -> str:
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
    )


def _call_github(prompt: str) -> str:
    """GitHub Models — free with a GitHub token."""
    return _call_openai_compatible(
        os.environ.get("GITHUB_MODELS_URL", "https://models.github.ai/inference/chat/completions"),
        GITHUB_MODELS_KEY,
        os.environ.get("GITHUB_MODELS_MODEL", "openai/gpt-4o-mini"),
        prompt,
    )


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
    if GEMINI_KEY:
        providers.append(("gemini", _call_gemini))
    if CEREBRAS_KEY:
        providers.append(("cerebras", _call_cerebras))
    if GROQ_KEY:
        providers.append(("groq", _call_groq))
    if OPENROUTER_KEY:
        providers.append(("openrouter", _call_openrouter))
    if GITHUB_MODELS_KEY:
        providers.append(("github", _call_github))

    if not providers:
        raise RuntimeError(
            "No LLM key configured. Set one of CEREBRAS_API_KEY, GEMINI_API_KEY, "
            "GROQ_API_KEY, OPENROUTER_API_KEY, or GITHUB_MODELS_TOKEN."
        )

    errors = []
    for name, fn in providers:
        for attempt in (1, 2):
            try:
                return fn(prompt)
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

    # Every configured provider failed.
    def _is_rate_limit(e):
        return isinstance(e, requests.exceptions.HTTPError) and getattr(e.response, "status_code", None) == 429

    if any(_is_rate_limit(e) for _, e in errors):
        raise LLMUnavailableError(
            "The AI model is receiving too many requests right now. "
            "Please wait a minute and try again."
        )
    if any(isinstance(e, (requests.exceptions.Timeout, requests.exceptions.ConnectionError,
                          requests.exceptions.HTTPError)) for _, e in errors):
        raise LLMUnavailableError(
            "The AI model is temporarily unavailable. Please try again in a moment."
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

    # Stage 5: Tavily demographics
    demos = _tavily_demographics(signals.get("title", ""), signals.get("location", ""))
    result["_pipeline"]["tavily_chars"] = len(demos)

    # Stage 6: Build prompt and call LLM
    prompt = _build_prompt(jd_content, signals, onet, wages, demos, fallback, market)
    raw = _call_llm(prompt)

    # Stage 7: Parse, validate, and normalize
    data = _parse_json(raw)
    data = validate_output(data)      # schema + safety check before it leaves the server
    data = _normalize_segments(data)
    data = _enforce_persona_count(data)   # deterministic, preset-aware count cap
    data = _scrub_internal_tokens(data)   # strip any leaked internal variable names

    # Guarantee the posted salary + location show even if the model omitted them.
    lc = data.get("local_context")
    if isinstance(lc, dict):
        if not lc.get("posted_compensation") and signals.get("salary"):
            lc["posted_compensation"] = signals["salary"]
        if not lc.get("metro_area") and signals.get("location"):
            lc["metro_area"] = signals["location"]

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
        return data

    # Production: strip every internal/debug field so nothing leaks to the browser.
    return {k: v for k, v in data.items() if not k.startswith("_")}
