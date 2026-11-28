"""
Serve the web viewer (``web/``) like ``python3 -m http.server``, but tell the browser to
check every file for changes before reusing it (``Cache-Control: no-cache``).

Without that header browsers may keep using an old copy of an edited ES module
(``src/*.js``) even after a normal reload, so a code change seems to have no effect.
Unchanged files still come back as a cheap 304 Not Modified.

    python3 web/serve.py [--bind 127.0.0.1] [port]      # default port 8000
"""

import argparse
import functools
import http.server
from pathlib import Path


class NoCacheHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("port", nargs="?", type=int, default=8000)
    ap.add_argument("--bind", default="127.0.0.1")
    args = ap.parse_args()
    handler = functools.partial(NoCacheHandler, directory=str(Path(__file__).resolve().parent))
    with http.server.ThreadingHTTPServer((args.bind, args.port), handler) as httpd:
        print(f"Serving web/ on http://{args.bind}:{args.port}/ (no-cache)")
        httpd.serve_forever()


if __name__ == "__main__":
    main()
