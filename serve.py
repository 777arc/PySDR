#!/usr/bin/env python3
"""Serve _build locally the way Cloudflare Pages serves pysdr.org.

The GNU Radio World embeds need SharedArrayBuffer, which browsers only hand to a
cross-origin isolated page.  In production the COOP/COEP headers that provide that
come from extra/_headers, but a plain `python -m http.server` sends neither, so
pressing Run inside an embed fails with "SharedArrayBuffer transfer requires
self.crossOriginIsolated" and nothing happens.  This sends the same two headers, so
the local preview behaves like the deployed site.

    python serve.py [port]     # defaults to 8091, the port PySDR uses locally

One local-only wrinkle: COEP blocks cross-origin subresources that don't opt in, and
a few images in the page templates are absolute https://pysdr.org/ URLs (the language
flags, for instance), so those don't render here.  They are same-origin in production.
"""

import sys
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer


class CrossOriginIsolatedHandler(SimpleHTTPRequestHandler):
    """Static file handler that adds the two headers extra/_headers sets in production."""

    def end_headers(self):
        self.send_header('Cross-Origin-Opener-Policy', 'same-origin')
        self.send_header('Cross-Origin-Embedder-Policy', 'require-corp')
        super().end_headers()


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8091
    handler = partial(CrossOriginIsolatedHandler, directory='_build')
    with ThreadingHTTPServer(('0.0.0.0', port), handler) as httpd:
        print('Serving _build with COOP/COEP at http://localhost:{}/'.format(port))
        print('The GNU Radio World embeds will actually run.  Ctrl-C to stop.')
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == '__main__':
    main()
