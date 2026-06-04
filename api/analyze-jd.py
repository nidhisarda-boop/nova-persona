import sys, os, json
import time
from collections import defaultdict, deque
sys.path.insert(0, os.path.dirname(__file__))
from _lib import build_persona_response, SafetyError
from http.server import BaseHTTPRequestHandler

# --- Rate limiting (in-memory, per warm instance) ---
RATE_LIMIT_MAX    = int(os.environ.get("RATE_LIMIT_MAX", "10"))     # requests
RATE_LIMIT_WINDOW = int(os.environ.get("RATE_LIMIT_WINDOW", "60"))  # seconds
_hits = defaultdict(deque)  # ip -> deque[timestamps]

def _rate_limited(ip):
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

            length = int(self.headers.get('Content-Length', 0))
            body   = json.loads(self.rfile.read(length) or b'{}')
            text   = body.get("text", "").strip()
            url    = body.get("url", "").strip()
            if not text and not url:
                self._json({"error": "Please provide a job posting URL or paste the job description text."}, 400)
                return
            result = build_persona_response(text=text, url=url)
            self._json(result)
        except SafetyError as e:
            # User-facing validation error — 400, not 500
            self._json({"error": str(e), "error_type": "input_validation"}, 400)
        except Exception as e:
            self._json({"error": str(e)}, 500)

    def _cors(self):
        self.send_header('Access-Control-Allow-Origin', '*')
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
