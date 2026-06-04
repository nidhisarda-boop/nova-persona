import sys, os, json
import time
from collections import defaultdict, deque
sys.path.insert(0, os.path.dirname(__file__))
import requests
from _lib import build_persona_response, SafetyError, OutputValidationError, LLMUnavailableError, SearchPageError
from http.server import BaseHTTPRequestHandler

# --- Rate limiting -----------------------------------------------------------
RATE_LIMIT_MAX    = int(os.environ.get("RATE_LIMIT_MAX", "10"))     # requests
RATE_LIMIT_WINDOW = int(os.environ.get("RATE_LIMIT_WINDOW", "60"))  # seconds

# Max accepted request body. Rejected before the body is read into memory.
MAX_BODY_BYTES = int(os.environ.get("MAX_BODY_BYTES", "100000"))    # 100 KB

# Comma-separated list of origins allowed to call this endpoint cross-origin.
ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "")

# Vercel KV / Upstash Redis REST (set automatically by the Vercel KV integration)
_KV_URL   = os.environ.get("KV_REST_API_URL")   or os.environ.get("UPSTASH_REDIS_REST_URL")
_KV_TOKEN = os.environ.get("KV_REST_API_TOKEN") or os.environ.get("UPSTASH_REDIS_REST_TOKEN")

# In-memory fallback (per warm instance) — used only when KV is not configured
_hits = defaultdict(deque)

def _rate_limited_memory(ip):
    now = time.time()
    q = _hits[ip]
    while q and q[0] <= now - RATE_LIMIT_WINDOW:
        q.popleft()
    if len(q) >= RATE_LIMIT_MAX:
        return True
    q.append(now)
    if len(_hits) > 10_000:
        for k in [k for k, v in _hits.items() if not v]:
            del _hits[k]
    return False

def _rate_limited_kv(ip):
    """Durable fixed-window counter in Redis. Returns True if over the limit.
    On any KV/network error, fail open to the in-memory limiter so a KV outage
    never takes the endpoint down."""
    try:
        bucket = int(time.time() // RATE_LIMIT_WINDOW)
        key = f"rl:{ip}:{bucket}"
        # Pipeline: INCR the counter, then set TTL only if not already set (NX).
        r = requests.post(
            f"{_KV_URL}/pipeline",
            headers={"Authorization": f"Bearer {_KV_TOKEN}"},
            json=[["INCR", key], ["EXPIRE", key, str(RATE_LIMIT_WINDOW * 2), "NX"]],
            timeout=2,
        )
        r.raise_for_status()
        count = int(r.json()[0]["result"])
        return count > RATE_LIMIT_MAX
    except Exception:
        return _rate_limited_memory(ip)

def _rate_limited(ip):
    if _KV_URL and _KV_TOKEN:
        return _rate_limited_kv(ip)
    return _rate_limited_memory(ip)

class handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_POST(self):
        try:
            # client IP: Vercel puts the real client first in X-Forwarded-For
            xff = self.headers.get('X-Forwarded-For', '')
            client_ip = xff.split(',')[0].strip() or self.headers.get('X-Real-IP', 'unknown')
            if _rate_limited(client_ip):
                self._json({
                    "error": "Too many requests. Please wait a minute and try again.",
                    "error_type": "rate_limited"
                }, 429)
                return

            # Reject oversized payloads before reading the body into memory
            length = int(self.headers.get('Content-Length', 0) or 0)
            if length > MAX_BODY_BYTES:
                self._json({
                    "error": "Request body is too large.",
                    "error_type": "payload_too_large"
                }, 413)
                return
            # Read at most the cap (+1 to detect an under-declared Content-Length)
            raw_body = self.rfile.read(min(length, MAX_BODY_BYTES + 1)) if length else b''
            if len(raw_body) > MAX_BODY_BYTES:
                self._json({
                    "error": "Request body is too large.",
                    "error_type": "payload_too_large"
                }, 413)
                return
            body   = json.loads(raw_body or b'{}')
            text   = body.get("text", "").strip()
            url    = body.get("url", "").strip()
            mode   = (body.get("mode") or "job_description").strip()
            if not text and not url:
                self._json({"error": "Please provide a job posting URL or paste the job description text."}, 400)
                return
            result = build_persona_response(text=text, url=url, mode=mode)
            self._json(result)
        except SearchPageError as e:
            # Not an error — a job-search/category page. Offer the user a choice.
            label = e.role_hint or "this role category"
            mkt = "" if e.market in ("", "Global") else f" ({e.market})"
            self._json({
                "page_type": "search_results",
                "role_hint": e.role_hint,
                "market": e.market,
                "message": "That link is a job-search results page, not a single job posting.",
                "options": [
                    {"id": "market_map", "label": f"Generate a general market map for “{label}”{mkt}"},
                    {"id": "pick_job", "label": "Open a specific job from the results and paste that link"},
                    {"id": "paste", "label": "Paste the full job description text"},
                ],
            }, 200)
        except SafetyError as e:
            # User-facing input validation error — 400
            self._json({"error": str(e), "error_type": "input_validation"}, 400)
        except OutputValidationError as e:
            # LLM produced malformed/unsafe output — 502, ask the user to retry
            self._json({"error": str(e), "error_type": "output_validation"}, 502)
        except LLMUnavailableError as e:
            # Upstream LLM rate-limited or down — 503 with a clear retry message
            self._json({"error": str(e), "error_type": "llm_unavailable"}, 503)
        except Exception as e:
            # Log the real error to the server (visible in Vercel logs only),
            # return a generic message so internal details never reach the client.
            import traceback
            traceback.print_exc()
            self._json({
                "error": "Something went wrong while generating personas. Please try again.",
                "error_type": "server_error"
            }, 500)

    def _cors(self):
        # Allowlist of origins permitted to call this endpoint cross-origin.
        # Comma-separated env var, e.g. "https://nova-persona.vercel.app,https://app.joveo.com".
        # When unset, no cross-origin header is sent — the app itself is served
        # same-origin so it keeps working, while other sites are blocked from
        # calling the endpoint and burning LLM credits.
        origin = self.headers.get('Origin', '')
        allowed = [o.strip() for o in ALLOWED_ORIGINS.split(',') if o.strip()]
        if allowed and origin in allowed:
            self.send_header('Access-Control-Allow-Origin', origin)
            self.send_header('Vary', 'Origin')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')

    def _json(self, data, status=200):
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self._cors()
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def log_message(self, *a):
        pass
