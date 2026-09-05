"""The pod's only way in. Stdlib only, and small enough to inline in dockerStartCmd.

No SSH, no git remote, no registry: the code arrives as a tarball PUT to this
server on :8788, and everything else about the boot is observable through it.
Keep it short - deploy.py embeds this file verbatim into the pod's start
command via a heredoc, and it must stay readable there.
"""
import json, os, signal, urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer

TOKEN = os.environ.get("UPLOAD_TOKEN", "")
# WS/RECEIVER_PORT exist so the unit test can run this file outside a pod.
WS = os.environ.get("WORKSPACE", "/workspace")
PORT = int(os.environ.get("RECEIVER_PORT", "8788"))
APP_PORT = int(os.environ.get("APP_PORT", "8787"))
TGZ, LOG, PHASE = WS + "/app.tgz", WS + "/boot.log", WS + "/phase"
PID = WS + "/server.pid"


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
