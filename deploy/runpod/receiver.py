"""The pod's only way in. Stdlib only, and small enough to inline in dockerStartCmd.

No SSH, no git remote, no registry: the code arrives as a tarball PUT to this
server on :8788, and everything else about the boot is observable through it.
Keep it short - deploy.py embeds this file verbatim into the pod's start
command via a heredoc, and it must stay readable there.
"""
import json, os, signal, tarfile, urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer

TOKEN = os.environ.get("UPLOAD_TOKEN", "")
# WS/RECEIVER_PORT exist so the unit test can run this file outside a pod.
WS = os.environ.get("WORKSPACE", "/workspace")
PORT = int(os.environ.get("RECEIVER_PORT", "8788"))
APP_PORT = int(os.environ.get("APP_PORT", "8787"))
TGZ, LOG, PHASE = WS + "/app.tgz", WS + "/boot.log", WS + "/phase"
PID = WS + "/server.pid"
#: A real tarball of apps/brushjam is ~300 KB. Anything this small is a probe,
#: a truncated transfer or a mistake - and accepting one used to cost a running
#: pod its server: the empty file replaced app.tgz and the SIGTERM killed a
#: bootstrap that was midway through uv sync.
MIN_UPLOAD_BYTES = 10 * 1024
#: What the boot loop runs after extracting. An archive without it cannot boot
#: the pod, so it is not an upload, whatever else it contains.
REQUIRED_MEMBER = "bootstrap.sh"


def looks_like_our_tarball(path):
    """Cheap gate, then the real one: gzip magic, then the member list.

    Validated before anything is replaced or signalled, so a bad upload costs
    the pod nothing at all.
    """
    try:
        with open(path, "rb") as f:
            if f.read(2) != b"\x1f\x8b":
                return "that is not a gzip file"
    except OSError as err:
        return "unreadable upload (%s)" % err
    try:
        with tarfile.open(path, "r:gz") as tar:
            for i, member in enumerate(tar):
                if member.name == REQUIRED_MEMBER:
                    return None
                if i > 20000:
                    break
    except Exception as err:
        return "that tarball did not open (%s)" % err
    return "the tarball has no %s, so it cannot boot the pod" % REQUIRED_MEMBER


def stop_server():
    """SIGTERM whatever bootstrap.sh last became, so the boot loop re-runs it.

    Not `pkill -f brushjam`: the boot loop's own command line contains this
    file, which contains that word, so pkill would kill the loop itself.
    """
    try:
        with open(PID) as f:
            os.kill(int(f.read().strip()), signal.SIGTERM)
    except Exception:
        pass


def healthy():
    try:
        with urllib.request.urlopen("http://127.0.0.1:%d/healthz" % APP_PORT, timeout=3) as r:
            return json.loads(r.read()).get("ok") is True
    except Exception:
        return False


class H(BaseHTTPRequestHandler):
    def reply(self, code, body, ctype="text/plain; charset=utf-8"):
        raw = body.encode() if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("content-type", ctype)
        self.send_header("content-length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def ok_token(self):
        from urllib.parse import parse_qs, urlparse
        return TOKEN and parse_qs(urlparse(self.path).query).get("token", [""])[0] == TOKEN

    def drain(self, n):
        """Read the body we are about to refuse.

        Replying without reading it resets the connection, and the client sees
        a socket error instead of the status we sent.
        """
        while n > 0:
            chunk = self.rfile.read(min(1 << 20, n))
            if not chunk:
                break
            n -= len(chunk)

    def do_PUT(self):
        raw = self.headers.get("content-length")
        try:
            n = int(raw)
            if n < 0:
                raise ValueError(raw)
        except (TypeError, ValueError):
            n = None
        if not self.path.startswith("/upload") or not self.ok_token():
            self.drain(n or 0)
            return self.reply(403, "forbidden")
        if n is None:
            # Distinct from "forbidden" on purpose: a chunked body and a bad
            # token are very different problems and used to look identical.
            return self.reply(
                411 if raw is None else 400,
                "the upload needs a valid Content-Length; a chunked body cannot be stored",
            )
        if n < MIN_UPLOAD_BYTES:
            # Refused before a single byte is written, so an empty PUT cannot
            # replace the installed tarball or stop the server.
            self.drain(n)
            return self.reply(400, "%d bytes is too small to be the app (min %d)" % (n, MIN_UPLOAD_BYTES))
        got = 0
        with open(TGZ + ".part", "wb") as f:
            while got < n:
                chunk = self.rfile.read(min(1 << 20, n - got))
                if not chunk:
                    break
                f.write(chunk)
                got += len(chunk)
        if got != n:
            os.remove(TGZ + ".part")
            return self.reply(400, "the upload was cut short (%d of %d bytes)" % (got, n))
        wrong = looks_like_our_tarball(TGZ + ".part")
        if wrong:
            os.remove(TGZ + ".part")
            return self.reply(400, wrong)
        os.replace(TGZ + ".part", TGZ)
        # The boot loop re-extracts the newer tarball once the server is down.
        stop_server()
        self.reply(200, "ok")

    def do_GET(self):
        if self.path.startswith("/log"):
            try:
                with open(LOG, "rb") as f:
                    return self.reply(200, b"".join(f.readlines()[-200:]))
            except OSError:
                return self.reply(200, "no log yet")
        if self.path.startswith("/status"):
            app = os.path.exists(WS + "/app/apps/brushjam/pyproject.toml")
            ckpt = os.environ.get("INPROC_CHECKPOINT", "")
            return self.reply(200, json.dumps({
                "phase": open(PHASE).read().strip() if os.path.exists(PHASE) else "starting",
                "app_present": app,
                "checkpoint_present": bool(ckpt) and os.path.exists(ckpt) and os.path.getsize(ckpt) > 6_000_000_000,
                "server_healthy": healthy(),
            }), "application/json")
        self.reply(404, "not found")

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    HTTPServer(("0.0.0.0", PORT), H).serve_forever()
