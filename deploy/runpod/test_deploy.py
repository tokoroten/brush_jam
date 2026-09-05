"""What can be checked without a pod: the tarball, the receiver, the start command.

    uv run --project apps/brushjam python deploy/runpod/test_deploy.py
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import socket
import sys
import tarfile
import tempfile
import threading
import urllib.request
from http.server import HTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import deploy as D  # noqa: E402

failures: list = []
CRLF = chr(13) + chr(10)


def check(name: str, condition: bool, detail: str = "") -> None:
    print(f"{'ok  ' if condition else 'FAIL'} {name}{'  ' + detail if detail and not condition else ''}")
    if not condition:
        failures.append(name)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def load_receiver(workspace: Path, port: int, token: str):
    os.environ.update(
        {"WORKSPACE": str(workspace), "RECEIVER_PORT": str(port), "UPLOAD_TOKEN": token, "APP_PORT": str(free_port())}
    )
    spec = importlib.util.spec_from_file_location("receiver_under_test", HERE / "receiver.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # __name__ != "__main__", so it does not serve
    return module


def test_tarball() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        dest = Path(tmp) / "app.tgz"
        D.build_tarball(dest)
        with tarfile.open(dest) as tar:
            names = tar.getnames()
        check("tarball has bootstrap.sh at the root", "bootstrap.sh" in names)
        check("tarball has pyproject", "apps/brushjam/pyproject.toml" in names)
        check("tarball has uv.lock", "apps/brushjam/uv.lock" in names)
        check("tarball has the built web client", "apps/brushjam/src/brushjam/static/index.html" in names)
        check("tarball has the pipeline", "apps/brushjam/src/brushjam/ai/pipeline.py" in names)
        with tarfile.open(dest) as tar:
            script = tar.extractfile("bootstrap.sh").read()
        check("bootstrap.sh is packed with LF endings", b"\r" not in script)
        check("bootstrap.sh is executable in the tarball", tarfile.open(dest).getmember("bootstrap.sh").mode == 0o755)
        check("tarball has no __pycache__", not any("__pycache__" in n for n in names))
        check("tarball has no .venv", not any(n.startswith("apps/brushjam/.venv") for n in names))
        check("tarball is under 5 MB", dest.stat().st_size < 5_000_000, f"{dest.stat().st_size} bytes")


def test_start_cmd() -> None:
    cmd = D.docker_start_cmd()
    script = cmd[2]
    check("start cmd is bash -lc", cmd[:2] == ["bash", "-lc"])
    check("receiver is inlined", "BaseHTTPRequestHandler" in script and "__RECEIVER_PY__" not in script)
    check("start cmd runs the receiver", "python3 /workspace/receiver.py" in script)
    check("start cmd runs bootstrap", "bash /workspace/app/bootstrap.sh" in script)
    check("start cmd carries no secret", "sk-" not in script and "hf_" not in script)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "start.sh"
        path.write_text(script, encoding="utf-8")
        bash = shutil.which("bash")
        if bash:
            import subprocess

            result = subprocess.run([bash, "-n", str(path)], capture_output=True, text=True)
            check("assembled start cmd parses", result.returncode == 0, result.stderr.strip())


def test_receiver() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        workspace = Path(tmp)
        port, token = free_port(), "test-token"
        receiver = load_receiver(workspace, port, token)
        server = HTTPServer(("127.0.0.1", port), receiver.H)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        base = f"http://127.0.0.1:{port}"
        try:
            payload = b"a tarball, more or less" * 1000

            def put(url: str) -> int:
                req = urllib.request.Request(url, data=payload, method="PUT")
                req.add_header("content-length", str(len(payload)))
                try:
                    with urllib.request.urlopen(req, timeout=10) as r:
                        return r.status
                except urllib.error.HTTPError as err:
                    return err.code

            check("PUT without a token is refused", put(f"{base}/upload") == 403)
            check("PUT with the wrong token is refused", put(f"{base}/upload?token=nope") == 403)
            check("PUT with the token is accepted", put(f"{base}/upload?token={token}") == 200)
            check("the upload landed", (workspace / "app.tgz").read_bytes() == payload)
            check("no partial file is left", not (workspace / "app.tgz.part").exists())

            with urllib.request.urlopen(f"{base}/status", timeout=10) as r:
                status = json.loads(r.read())
            check("status reports the phase", status["phase"] == "starting")
            check("status knows the app is absent", status["app_present"] is False)
            check("status knows the checkpoint is absent", status["checkpoint_present"] is False)
            check("status knows the server is down", status["server_healthy"] is False)

            with urllib.request.urlopen(f"{base}/log", timeout=10) as r:
                check("log before any boot", r.read() == b"no log yet")
            (workspace / "boot.log").write_text("\n".join(f"line {i}" for i in range(500)), encoding="utf-8")
            with urllib.request.urlopen(f"{base}/log", timeout=10) as r:
                lines = r.read().decode().splitlines()
            check("log returns the last 200 lines", len(lines) == 200 and lines[-1] == "line 499")

            (workspace / "phase").write_text("checkpoint\n", encoding="utf-8")
            with urllib.request.urlopen(f"{base}/status", timeout=10) as r:
                check("status follows the phase file", json.loads(r.read())["phase"] == "checkpoint")

            try:
                urllib.request.urlopen(f"{base}/secrets", timeout=10)
                check("unknown paths 404", False)
            except urllib.error.HTTPError as err:
                check("unknown paths 404", err.code == 404)

            # The way deploy.py actually uploads. A file-object body made
            # urllib send it chunked, and that failed with a bare 403.
            (workspace / "app.tgz").unlink()
            tarball = workspace / "local.tgz"
            tarball.write_bytes(payload)
            D.proxy = lambda pod_id, port: base  # type: ignore[assignment]
            D.upload("pod", token, tarball)
            check("deploy.upload lands the tarball", (workspace / "app.tgz").read_bytes() == payload)

            def raw_put(headers: list, body: bytes, half_close: bool = False) -> str:
                """A PUT the http client libraries will not let us build."""
                head = CRLF.join([f"PUT /upload?token={token} HTTP/1.1", "host: x", *headers, "", ""])
                with socket.create_connection(("127.0.0.1", port), timeout=10) as sock:
                    sock.sendall(head.encode() + body)
                    if half_close:  # tell the server no more body is coming
                        sock.shutdown(socket.SHUT_WR)
                    data = b""
                    while (CRLF + CRLF).encode() not in data:
                        more = sock.recv(4096)
                        if not more:
                            break
                        data += more
                    return data.decode("utf-8", "replace")

            close = ["connection: close"]
            check("no Content-Length is 411, not 403", " 411 " in raw_put(close, b""))
            chunked = raw_put(["transfer-encoding: chunked", *close], ("0" + CRLF + CRLF).encode())
            check("a chunked body is 411, not 403", " 411 " in chunked)
            check("a bad Content-Length is 400", " 400 " in raw_put(["content-length: nope", *close], b""))
            reply = raw_put(["content-length: 100", *close], b"short", half_close=True)
            check("a cut-short body is refused", " 400 " in reply)
            check("a cut-short body leaves no partial file", not (workspace / "app.tgz.part").exists())
            check("a cut-short body does not replace the tarball", (workspace / "app.tgz").read_bytes() == payload)

            (workspace / "server.pid").write_text("999999999", encoding="utf-8")
            receiver.stop_server()  # a stale pid must not raise
            check("stop_server survives a stale pid", True)
        finally:
            server.shutdown()


def test_watch_boot() -> None:
    """The deploy has to show progress by itself; `log` is for detail."""
    replies = [
        json.dumps({"phase": "deps", "app_present": True, "checkpoint_present": False, "server_healthy": False}),
        json.dumps({"phase": "deps", "app_present": True, "checkpoint_present": False, "server_healthy": False}),
        json.dumps({"phase": "checkpoint", "app_present": True, "checkpoint_present": False, "server_healthy": False}),
        json.dumps({"phase": "server", "app_present": True, "checkpoint_present": True, "server_healthy": True}),
    ]
    printed: list = []
    real_get, real_sleep, real_print = D.get, D.time.sleep, print
    D.get = lambda url, timeout=20: replies.pop(0)
    D.time.sleep = lambda _s: None
    import builtins

    builtins.print = lambda *a, **k: printed.append(" ".join(str(x) for x in a))
    try:
        healthy = D.watch_boot("pod", 60)
    finally:
        D.get, D.time.sleep, builtins.print = real_get, real_sleep, real_print
    check("watch_boot ends when the server is healthy", healthy is True)
    check("watch_boot prints each phase once", sum("deps" in line for line in printed) == 1, str(printed))
    check("watch_boot prints the later phases", any("checkpoint" in line for line in printed))

    D.get = lambda url, timeout=20: (_ for _ in ()).throw(OSError("no route"))
    D.time.sleep = lambda _s: None
    builtins.print = lambda *a, **k: None
    try:
        timed_out = D.watch_boot("pod", -1)
    finally:
        D.get, D.time.sleep, builtins.print = real_get, real_sleep, real_print
    check("watch_boot gives up on the timeout", timed_out is False)


if __name__ == "__main__":
    test_tarball()
    test_start_cmd()
    test_receiver()
    test_watch_boot()
    print()
    if failures:
        print(f"{len(failures)} failed: {', '.join(failures)}")
        raise SystemExit(1)
    print("all checks passed")
