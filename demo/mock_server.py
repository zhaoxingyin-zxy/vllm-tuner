"""
demo/mock_server.py
-------------------
Minimal HTTP server that mimics the vLLM OpenAI-compatible API.
Used for local integration testing without a real GPU.

Endpoints implemented:
  GET  /health                  → 200 {"status": "ok"}
  GET  /v1/models               → 200 (LMDeploy compat)
  POST /v1/chat/completions     → 200 with mock completion

Usage:
  python demo/mock_server.py [--port 8000]
"""
import argparse
import json
import time
import random
from http.server import BaseHTTPRequestHandler, HTTPServer


MOCK_MODEL = "mock-llm"
WORDS = ["The", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"]


class MockVLLMHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Suppress default request logging (too noisy for tests)
        pass

    def _send_json(self, status: int, body: dict):
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path in ("/health", "/v1/health"):
            self._send_json(200, {"status": "ok"})
        elif self.path == "/v1/models":
            self._send_json(200, {"data": [{"id": MOCK_MODEL}]})
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self):
        if self.path == "/v1/chat/completions":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length else {}
            max_tokens = body.get("max_tokens", 32)

            # Simulate a short latency (5-15ms)
            time.sleep(random.uniform(0.005, 0.015))

            # Generate a mock completion
            n_tokens = min(max_tokens, random.randint(8, 20))
            content = " ".join(random.choices(WORDS, k=n_tokens))

            self._send_json(200, {
                "id": "mock-cmpl-001",
                "object": "chat.completion",
                "model": MOCK_MODEL,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }],
                "usage": {
                    "prompt_tokens": len(body.get("messages", [{}])[-1].get("content", "").split()),
                    "completion_tokens": n_tokens,
                    "total_tokens": n_tokens + 10,
                },
            })
        else:
            self._send_json(404, {"error": "not found"})


def run(port: int = 8000):
    server = HTTPServer(("0.0.0.0", port), MockVLLMHandler)
    print(f"[MockServer] Listening on http://localhost:{port}")
    print(f"[MockServer] Health: http://localhost:{port}/health")
    print(f"[MockServer] Chat:   http://localhost:{port}/v1/chat/completions")
    print("[MockServer] Ctrl-C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[MockServer] Stopped.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mock vLLM server for local testing")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    run(args.port)
