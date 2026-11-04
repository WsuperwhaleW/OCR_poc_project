"""Web plumbing shared by every Flask app of this project.

Split out of app.py on 2026-11-04 (worker-service plan, step 2): the benchmark
(`app.py`) and the worker service (`service_app.py`) both mount under a
reverse-proxy sub-path, put each request in its browser tab's workspace,
gzip text replies and reject non-object JSON bodies. Moved verbatim.
"""
import gzip
import re
import zlib

from flask import jsonify, request

import workspace


class ProxyPrefix:
    """Answer a request made under any path prefix as if it were made at `/`.

    **Worked out per request, never configured** (2026-10-20): the app sits
    behind a reverse proxy at a sub-path -- nginx serving `host:3000/OCR/` --
    and the page builds every request from its own address, so whatever path
    the page was reached at is the prefix. A proxy that strips the prefix
    (`proxy_pass http://host:port/;`) sends plain `/api/...` and nothing here
    happens. One that passes it through sends `/OCR/api/...`, and:

    * `X-Forwarded-Prefix`, where the proxy sends one, is taken first;
    * otherwise everything before `/api/` is the prefix -- every route but the
      page itself lives under `/api/`;
    * otherwise a GET for a path that is no route of this app and ends in `/`
      (or has no dot in its last part) is the page, served at that prefix.

    The prefix moves to SCRIPT_NAME, where Flask expects a mount point.
    """

    def __init__(self, flask_app):
        self.flask_app = flask_app
        self.wsgi = flask_app.wsgi_app

    def _is_route(self, path: str, method: str) -> bool:
        try:
            self.flask_app.url_map.bind("localhost").match(path, method=method)
            return True
        except Exception:                           # NotFound, MethodNotAllowed, redirects
            return False

    def _prefix(self, environ, path: str) -> str:
        forwarded = (environ.get("HTTP_X_FORWARDED_PREFIX") or "").strip().rstrip("/")
        if forwarded and (path == forwarded or path.startswith(forwarded + "/")):
            return forwarded
        method = environ.get("REQUEST_METHOD", "GET")
        if self._is_route(path, method):
            return ""
        # Match an actual nested route before falling back to the legacy mount
        # heuristic: /ocr/user and /ocr/static/user/app.js are not the main page.
        for match in re.finditer('/', path[1:]):
            at = match.start() + 1
            if self._is_route(path[at:], method):
                return path[:at]
        at = path.find("/api/")
        if at > 0:
            return path[:at]
        last = path.rstrip("/").rsplit("/", 1)[-1]
        if (method in ("GET", "HEAD") and path != "/" and not self._is_route(path, method)
                and (path.endswith("/") or "." not in last)):
            return path.rstrip("/")
        return ""

    def __call__(self, environ, start_response):
        path = environ.get("PATH_INFO", "") or "/"
        prefix = self._prefix(environ, path)
        if prefix:
            environ["SCRIPT_NAME"] = environ.get("SCRIPT_NAME", "") + prefix
            environ["PATH_INFO"] = path[len(prefix):] or "/"
        return self.wsgi(environ, start_response)


def validate_api_json_object():
    """JSON API bodies are objects; reject malformed/scalar input before handlers.

    Leave multipart uploads and bodyless requests alone. Some endpoints use an
    omitted body to select their saved defaults.
    """
    if (request.path.startswith('/api/')
            and request.method in ('POST', 'PUT', 'PATCH', 'DELETE')
            and request.is_json
            and not isinstance(request.get_json(silent=True), dict)):
        return jsonify(error='Request body must be a valid JSON object.'), 400


COMPRESSIBLE = ("text/", "application/json", "application/javascript",
                "application/wasm", "application/x-ndjson", "image/svg+xml")


def enter_workspace():
    """Put this request in its browser tab's workspace -- see `workspace`.

    The page sends its id as a header on every fetch, and as `?ws=` on the page
    load itself, which no script can add a header to. Neither: the default
    workspace, which is how every caller other than the page has always run.
    """
    workspace.enter(request.headers.get(workspace.HEADER) or request.args.get("ws"))


def _gzip_stream(chunks):
    """Gzip a streamed reply, flushing per chunk so each still arrives at once."""
    packer = zlib.compressobj(6, zlib.DEFLATED, 31)
    try:
        for chunk in chunks:
            chunk = chunk.encode("utf-8") if isinstance(chunk, str) else chunk
            if chunk:
                yield packer.compress(chunk) + packer.flush(zlib.Z_SYNC_FLUSH)
        yield packer.flush()
    finally:
        # The stream's own `finally` is what releases a sweep guard.
        close = getattr(chunks, "close", None)
        if close:
            close()


def gzip_response(response):
    """Compress a text reply for a client that accepts it.

    Added 2026-10-21 for JSON, at the user's request: *save time and data
    transfer of log retriving*. Widened on 2026-11-02 because outbound transfer
    is what the server is billed for: the page's HTML, scripts and styles
    compress 3-4x and the ndjson reads were not compressed at all. A stream is
    flushed chunk by chunk, so it still reaches the page line by line. Images
    and PDFs are not worth compressing, and anything already encoded is left
    alone, so a proxy in front can still compress whatever this does not.
    """
    if (response.status_code not in (200, 201, 202)
            or not (response.mimetype or "").startswith(COMPRESSIBLE)
            or "Content-Encoding" in response.headers):
        return response
    response.vary.add("Accept-Encoding")
    if "gzip" not in request.headers.get("Accept-Encoding", "").lower():
        return response
    if response.is_streamed and not response.direct_passthrough:
        response.response = _gzip_stream(response.response)
        response.headers.pop("Content-Length", None)
    else:
        response.direct_passthrough = False
        body = response.get_data()
        if len(body) < 1024:
            return response
        response.set_data(gzip.compress(body, compresslevel=6))
    response.headers["Content-Encoding"] = "gzip"
    # A byte range of the file means nothing against the compressed body.
    response.headers.pop("Accept-Ranges", None)
    etag, weak = response.get_etag()
    if etag and not weak:
        response.set_etag(etag, weak=True)
    return response


def install(flask_app, max_upload_mb):
    """The web plumbing every app of this project shares, in its original order:
    upload cap, sub-path mount, per-tab workspace, gzip. Call before registering
    routes; register `validate_api_json_object` after the User routes."""
    # The file cap excludes the multipart envelope. request_input checks the bytes
    # of the actual upload; a file exactly at the advertised limit must still fit.
    flask_app.config["MAX_CONTENT_LENGTH"] = (max_upload_mb + 1) * 1024 * 1024
    flask_app.wsgi_app = ProxyPrefix(flask_app)
    flask_app.before_request(enter_workspace)
    flask_app.after_request(gzip_response)
