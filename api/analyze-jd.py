import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))
from _lib import build_persona_response
from http.server import BaseHTTPRequestHandler

class handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_POST(self):
        try:
            length = int(self.headers.get('Content-Length', 0))
            body   = json.loads(self.rfile.read(length) or b'{}')
            text   = body.get("text", "").strip()
            url    = body.get("url", "").strip()
            if not text and not url:
                self._json({"error": "Provide 'text' or 'url'"}, 400)
                return
            result = build_persona_response(text=text, url=url)
            self._json(result)
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
