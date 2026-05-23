from __future__ import annotations

import argparse
import http.client
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parent


class DemoProxyHandler(BaseHTTPRequestHandler):
    backend_host = "127.0.0.1"
    backend_port = 8120

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/":
            path = "/index.html"
        target = (ROOT / path.lstrip("/")).resolve()
        if not target.is_file() or ROOT not in target.parents:
            self.send_error(404)
            return

        data = target.read_bytes()
        content_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/chat":
            self.send_error(404)
            return

        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        conn = http.client.HTTPConnection(self.backend_host, self.backend_port, timeout=120)
        try:
            conn.request(
                "POST",
                "/api/chat",
                body=body,
                headers={"Content-Type": self.headers.get("Content-Type", "application/json")},
            )
            response = conn.getresponse()
            data = response.read()
        finally:
            conn.close()

        self.send_response(response.status)
        self.send_header("Content-Type", response.getheader("Content-Type", "application/json"))
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5174)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), DemoProxyHandler)
    print(f"Serving playable demo proxy on http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
