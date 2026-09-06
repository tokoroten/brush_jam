"""What can be checked without a pod: the tarball, the receiver, the start command.

    uv run --project apps/brushjam python deploy/runpod/test_deploy.py
"""

from __future__ import annotations

import importlib.util
import io
import json
import os
import shutil
import socket
import sys
import tarfile
import time
import tempfile
import threading
import urllib.request
from http.server import HTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import deploy as D  # noqa: E402
from deploy import IdleTracker, watch_loop  # noqa: E402

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
    check("start cmd runs the receiver", 'python3 "$WS/receiver.py"' in script)
    check("start cmd runs bootstrap", 'bash "$WS/app/bootstrap.sh"' in script)
    check("start cmd claims an upload before extracting", 'mv "$WS/app.tgz" "$claim"' in script)
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
            # A real one: the receiver refuses anything that is not the app,
            # so the happy path has to send the genuine article.
            payload = D.build_tarball(workspace / "upload.tgz").read_bytes()

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

            # An empty or tiny PUT used to be accepted: it replaced app.tgz
            # and SIGTERMed the server, which killed a bootstrap that was
            # midway through uv sync and left the pod with nothing to run.
            def put_body(body: bytes) -> int:
                req = urllib.request.Request(f"{base}/upload?token={token}", data=body, method="PUT")
                req.add_header("content-length", str(len(body)))
                try:
                    with urllib.request.urlopen(req, timeout=10) as r:
                        return r.status
                except urllib.error.HTTPError as err:
                    return err.code

            check("an empty upload is refused", put_body(b"") == 400)
            check("a tiny upload is refused", put_body(b"x" * 4096) == 400)
            not_gzip = bytes(20000)
            check("a body that is not gzip is refused", put_body(not_gzip) == 400)

            wrong = io.BytesIO()
            with tarfile.open(fileobj=wrong, mode="w:gz") as tar:
                info = tarfile.TarInfo("some/other/file.txt")
                info.size = 20000
                tar.addfile(info, io.BytesIO(bytes(20000)))
            check("a tarball without bootstrap.sh is refused", put_body(wrong.getvalue()) == 400)

            check("none of that replaced the installed tarball", (workspace / "app.tgz").read_bytes() == payload)
            check("none of that left a partial file", not (workspace / "app.tgz.part").exists())

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


def test_user_agent() -> None:
    """Every proxied call has to name itself.

    The RunPod proxy answers 403 to urllib's default user agent, which is
    indistinguishable from a rejected upload token at the call site - it cost
    two rounds of debugging the token instead of the header.
    """
    req = D.proxy_request("https://pod-8788.proxy.runpod.net/status")
    check("a proxied GET carries a user agent", req.get_header("User-agent") == D.USER_AGENT)
    check("it is not urllib's default", "urllib" not in (req.get_header("User-agent") or ""))

    put = D.proxy_request(
        "https://pod-8788.proxy.runpod.net/upload?token=x",
        data=b"body",
        method="PUT",
        content_type="application/octet-stream",
        content_length="4",
    )
    check("an upload carries it too", put.get_header("User-agent") == D.USER_AGENT)
    check("the upload is still a PUT", put.get_method() == "PUT")
    check("the upload still declares its length", put.get_header("Content-length") == "4")

    # get() is what /status, /log and the /healthz poll all go through, so
    # every proxied read is covered by it carrying the header.
    seen: list = []
    real = D.urllib.request.urlopen

    class Fake:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b"ok"

    D.urllib.request.urlopen = lambda request, timeout=None: (seen.append(request), Fake())[1]
    try:
        D.get("https://pod-8787.proxy.runpod.net/healthz")
    finally:
        D.urllib.request.urlopen = real
    check("get() sends the header", seen and seen[0].get_header("User-agent") == D.USER_AGENT)


def test_phase_contract() -> None:
    """bootstrap.sh writes the phase; the receiver reads it. They must agree.

    The receiver is written into the pod by dockerStartCmd and outlives every
    upload, so a bootstrap that changed how it reports progress would leave the
    pod claiming "starting" for the whole boot with no way to notice.
    """
    bash = shutil.which("bash")
    if not bash:
        check("phase contract (bash unavailable)", True)
        return
    import subprocess

    with tempfile.TemporaryDirectory() as tmp:
        workspace = Path(tmp)
        port, token = free_port(), "phase-token"
        receiver = load_receiver(workspace, port, token)

        posix = workspace.as_posix()
        if len(posix) > 1 and posix[1] == ":":
            posix = "/" + posix[0].lower() + posix[2:]

        written = []
        for name in ("uv", "deps", "checkpoint", "server"):
            script = (
                f'BOOTSTRAP_FUNCTIONS_ONLY=1 WORKSPACE_DIR="{posix}" '
                f'. "{(HERE / "bootstrap.sh").as_posix()}"; phase "{name}"'
            )
            subprocess.run([bash, "-c", script], capture_output=True, text=True, check=True)
            # Read it the way the receiver does, through the receiver.
            written.append((name, json.loads(status_of(receiver))["phase"]))
        check("the receiver reports the phase bootstrap wrote", all(a == b for a, b in written), str(written))

        # start.sh owns the phases before and between bootstrap runs.
        start = (HERE / "start.sh").read_text(encoding="utf-8")
        for name in ("waiting-for-upload", "extract", "restarting"):
            check(f'start.sh writes the phase "{name}"', f'echo "{name}" > "$WS/phase"' in start)
        bootstrap = (HERE / "bootstrap.sh").read_text(encoding="utf-8")
        check("both write to <workspace>/phase", '"$WORKSPACE/phase"' in bootstrap and '"$WS/phase"' in start)
        check("the venv is outside the swapped tree", "UV_PROJECT_ENVIRONMENT" in bootstrap)
        check("the venv is not inside /workspace/app", "/app/" not in bootstrap.split("UV_PROJECT_ENVIRONMENT")[1][:80])


def status_of(receiver) -> str:
    """The receiver's own /status body, without a socket."""
    import io

    class Probe(receiver.H):
        def __init__(self) -> None:  # no socket, no request line
            self.path = "/status"
            self.body = b""

        def reply(self, code, body, ctype="text/plain; charset=utf-8"):
            self.body = body.encode() if isinstance(body, str) else body

    probe = Probe()
    probe.do_GET()
    return probe.body.decode()


def test_install_pending() -> None:
    """The boot loop's claim-and-extract step, driven directly in bash.

    The bug this pins down: an upload that arrives while the previous one is
    being extracted was acknowledged by the receiver and then never installed,
    because the loop compared the tarball's mtime with a stamp written after
    extraction finished. Nothing here looks at a timestamp - the file's
    existence is the pending flag and renaming it is the claim.
    """
    bash = shutil.which("bash")
    if not bash:
        check("install_pending (bash unavailable)", True)
        return
    import subprocess

    with tempfile.TemporaryDirectory() as tmp:
        ws = Path(tmp)
        # GNU tar reads "C:/x" as a remote host, so the bash side gets the
        # POSIX form of the path this shell would use.
        posix = ws.as_posix()
        if len(posix) > 1 and posix[1] == ":":
            posix = "/" + posix[0].lower() + posix[2:]

        def make_tarball(marker: str) -> None:
            payload = ws / "payload.txt"
            payload.write_text(marker, encoding="utf-8")
            with tarfile.open(ws / "app.tgz", "w:gz") as tar:
                tar.add(payload, arcname="marker.txt")
            payload.unlink()

        def install() -> int:
            script = f'START_SH_FUNCTIONS_ONLY=1 WORKSPACE_DIR="{posix}" . "{(HERE / "start.sh").as_posix()}"; install_pending'
            return subprocess.run([bash, "-c", script], capture_output=True, text=True).returncode

        check("nothing pending is not an install", install() != 0)

        make_tarball("A")
        check("a pending tarball installs", install() == 0)
        check("the tarball was claimed", not (ws / "app.tgz").exists())
        check("A is installed", (ws / "app" / "marker.txt").read_text() == "A")

        # B lands during A's extraction: older than anything the loop wrote,
        # and the only reason it is pending is that the file is there.
        make_tarball("B")
        os.utime(ws / "app.tgz", (0, 0))
        check("an upload that raced the extraction still installs", install() == 0)
        check("B replaced A", (ws / "app" / "marker.txt").read_text() == "B")

        # A tarball that is not one must not wipe the running install.
        (ws / "app.tgz").write_bytes(b"not a tarball")
        check("a corrupt tarball is refused", install() != 0)
        check("the previous install survives it", (ws / "app" / "marker.txt").read_text() == "B")
        check("the corrupt tarball is cleared", not (ws / "app.tgz").exists())


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


def test_idle_watch() -> None:
    """`watch` decides on two numbers from /healthz, on a fake clock.

    Nothing here talks to a pod: the poller and both clocks are injected, so
    thirty idle minutes take no time and the awkward cases - a socket that
    connects at minute 29, a server too old to report anything - are ordinary
    assertions.
    """
    clock = {"t": 1_000_000.0}
    slept: list = []

    def now() -> float:
        return clock["t"]

    def sleep(seconds: float) -> None:
        slept.append(seconds)
        clock["t"] += seconds

    def health(sockets=0, generation=None, generations=0):
        return {
            "ok": True,
            "backend": "inproc",
            "rooms": 1,
            "active_sockets": sockets,
            "last_generation_at": generation,
            "generations": generations,
        }

    def run(replies, idle_seconds=1800, interval=60):
        """Answer with each reply in turn, then repeat the last one forever."""
        printed: list = []
        queue = list(replies)
        last = {"value": queue[-1] if queue else None}

        def poll():
            if queue:
                last["value"] = queue.pop(0)
            return last["value"]

        reason = watch_loop(
            poll,
            idle_seconds=idle_seconds,
            interval=interval,
            now=now,
            sleep=sleep,
            log=printed.append,
        )
        return reason, printed

    # An empty pod: thirty minutes of nothing, then a stop.
    clock["t"] = 1_000_000.0
    reason, printed = run([health()])
    check("watch stops an idle pod", "for 30 minutes" in reason, reason)
    check("watch waited the full window", clock["t"] - 1_000_000.0 >= 1800, str(clock["t"]))
    check("watch polls once a minute", slept and set(slept) == {60}, str(set(slept)))
    check("watch prints the state once, not every poll", len(printed) <= 3, str(printed))

    # Somebody is in the room the whole time: the loop never returns, so it is
    # driven by hand rather than by watch_loop.
    tracker = IdleTracker(1800)
    at = 2_000_000.0
    for _ in range(60):
        tracker.observe(health(sockets=2), at)
        check_once("a connected socket keeps the pod alive", tracker.stop_reason(at) is None)
        at += 60
    check("a busy pod is never stopped", tracker.stop_reason(at) is None)

    # A socket at minute 29 resets the clock.
    tracker = IdleTracker(1800)
    at = 3_000_000.0
    tracker.observe(health(), at)
    at += 29 * 60
    tracker.observe(health(), at)
    check("still not stopped at 29 minutes", tracker.stop_reason(at) is None)
    tracker.observe(health(sockets=1), at)
    at += 29 * 60
    tracker.observe(health(), at)
    check("a visitor resets the idle clock", tracker.stop_reason(at) is None, tracker.describe(at))
    at += 31 * 60
    tracker.observe(health(), at)
    check("and it stops 30 minutes after they leave", tracker.stop_reason(at) is not None)

    # A generation between two polls is activity even with nobody connected -
    # a tool driving the room over HTTP, or a settings change mid-generation.
    tracker = IdleTracker(1800)
    at = 4_000_000.0
    tracker.observe(health(generation=None), at)
    at += 31 * 60
    tracker.observe(health(generation=int(at * 1000) - 10_000, generations=1), at)
    check("a generation counts as activity", tracker.stop_reason(at) is None, tracker.describe(at))

    # A generation older than the whole window does not.
    tracker = IdleTracker(1800)
    at = 5_000_000.0
    stale = int((at - 3600) * 1000)
    tracker.observe(health(generation=stale, generations=4), at)
    at += 31 * 60
    tracker.observe(health(generation=stale, generations=4), at)
    check("an old generation does not keep it alive", tracker.stop_reason(at) is not None)

    # Review 3 finding 6: a generation used to keep the pod "busy" for the
    # whole window, and only then did the idle clock start - so the default
    # 30-minute watch stopped an hour after the last picture. The deadline is
    # the last activity plus the window, not the first idle poll plus it.
    tracker = IdleTracker(1800)
    at = 4_500_000.0
    generated = int(at * 1000)
    tracker.observe(health(generation=generated, generations=1), at)
    check("a generation just now is busy", tracker.stop_reason(at) is None)
    at += 29 * 60
    tracker.observe(health(generation=generated, generations=1), at)
    check("...still busy at 29 minutes", tracker.stop_reason(at) is None, tracker.describe(at))
    at += 2 * 60
    tracker.observe(health(generation=generated, generations=1), at)
    check(
        "...and stopped at 30, not 60",
        tracker.stop_reason(at) is not None,
        tracker.describe(at),
    )

    # The same through the loop, which is what the pod actually runs: one
    # generation at t=0, nobody connected, stopped half an hour later.
    clock["t"] = 4_600_000.0
    started = clock["t"]
    reason, _ = run([health(generation=int(started * 1000), generations=1)])
    waited = clock["t"] - started
    check("the loop stops one window after the generation", reason is not None, str(reason))
    check(
        "...which is 30 minutes, not 60",
        1800 <= waited < 1800 + 120,
        "%d s" % waited,
    )

    # A watch started beside a pod that has been idle for hours does not grant
    # it another window for having been idle.
    tracker = IdleTracker(1800)
    at = 4_700_000.0
    tracker.observe(health(generation=int((at - 4 * 3600) * 1000), generations=9), at)
    check("a long-idle pod is stopped at the first poll", tracker.stop_reason(at) is not None)

    # A server that does not report activity is never stopped.
    tracker = IdleTracker(1800)
    at = 6_000_000.0
    for _ in range(40):
        tracker.observe({"ok": True, "backend": "inproc", "rooms": 0}, at)
        at += 60
    check("an old server is left running", tracker.stop_reason(at) is None, tracker.describe(at))
    check("...and says why", "does not report activity" in tracker.detail, tracker.detail)

    # A pod that never answers is stopped too - that is money burning with
    # nothing to show for it - and the reason says so.
    clock["t"] = 7_000_000.0
    reason, _ = run([None])
    check("an unreachable pod is stopped", "has not answered" in reason, reason)

    # Only state changes are printed.
    clock["t"] = 8_000_000.0
    reason, printed = run([health(sockets=1)] + [health()] * 40, idle_seconds=600)
    check("the change from busy to idle is printed", len(printed) == 2, str(printed))
    check("busy is described with its sockets", "1 socket connected" in printed[0], str(printed))


_once: set = set()


def check_once(name: str, condition: bool) -> None:
    """A check inside a loop: report it once, unless it fails."""
    if not condition or name not in _once:
        _once.add(name)
        check(name, condition)


def test_watch_ctrl_c_leaves_the_pod_alone() -> None:
    """Ctrl+C is not "stop the pod": it is "stop watching"."""
    import argparse
    import builtins

    calls: list = []
    printed: list = []
    real = (D.read_pod, D.api, D.watch_loop, D.load_env, builtins.print)
    D.read_pod = lambda: {"id": "pod", "token": "t"}
    D.load_env = lambda: {}
    D.api = lambda *a, **k: calls.append(a)
    D.watch_loop = lambda *a, **k: (_ for _ in ()).throw(KeyboardInterrupt())
    builtins.print = lambda *a, **k: printed.append(" ".join(str(x) for x in a))
    try:
        D.cmd_watch(argparse.Namespace(idle_minutes=30, interval=60, terminate=False))
    finally:
        D.read_pod, D.api, D.watch_loop, D.load_env, builtins.print = real
    check("Ctrl+C stops nothing", calls == [], str(calls))
    check("...and says so", any("untouched" in line for line in printed), str(printed))


def test_watch_stops_rather_than_terminates() -> None:
    import argparse
    import builtins

    for terminate, expect in ((False, "POST"), (True, "DELETE")):
        calls: list = []
        real = (D.read_pod, D.api, D.watch_loop, D.load_env, D.POD_FILE, builtins.print)
        D.read_pod = lambda: {"id": "pod", "token": "t"}
        D.load_env = lambda: {}
        D.api = lambda env, method, path, body=None: calls.append((method, path))
        D.watch_loop = lambda *a, **k: "nobody has been connected"
        D.POD_FILE = Path(tempfile.gettempdir()) / "brushjam-test-nonexistent.pod"
        builtins.print = lambda *a, **k: None
        try:
            D.cmd_watch(argparse.Namespace(idle_minutes=30, interval=60, terminate=terminate))
        finally:
            D.read_pod, D.api, D.watch_loop, D.load_env, D.POD_FILE, builtins.print = real
        check(
            "watch %s the pod" % ("terminates" if terminate else "stops"),
            calls == [(expect, "/pods/pod" + ("" if terminate else "/stop"))],
            str(calls),
        )


def test_supervisor_swaps_a_running_server() -> None:
    """Review 2 finding 7: an upload that arrives while the server is running.

    The receiver SIGTERMs the pid bootstrap.sh recorded, but an upload landing
    before that pid is published - or after a stale one was signalled - reached
    nobody: the loop blocked on the old server in the foreground and the new
    code sat in /workspace/app.tgz forever. The supervisor now runs the server
    as a child and watches for both its exit and a pending tarball.
    """
    import subprocess

    bash = _bash()
    if bash is None:  # pragma: no cover - a machine with no usable bash
        print("skip supervisor swap (no working bash)")
        return

    fake_server = (
        'echo $$ > "$WORKSPACE_DIR/server.pid"\n'
        'echo {tag} >> "$WORKSPACE_DIR/ran"\n'
        "while true; do sleep 0.2; done\n"
    )
    script = D.docker_start_cmd()[2].replace(
        "__RECEIVER_PY__", "print('no receiver in this test')"
    )
    with tempfile.TemporaryDirectory() as tmp:
        ws = Path(tmp)
        (ws / "app").mkdir()
        # v1: a "server" that runs until it is stopped, as the real one does.
        (ws / "app" / "bootstrap.sh").write_text(
            fake_server.format(tag="v1"), encoding="utf-8", newline="\n"
        )
        start = ws / "start.sh"
        start.write_text(script, encoding="utf-8", newline="\n")

        env = dict(os.environ)
        env.update(
            {
                # As the shell inside sees it: msys tar reads "C:/x" as a
                # remote host and refuses to extract there.
                "WORKSPACE_DIR": _bash_path(ws),
                "WORKSPACE": _bash_path(ws),
                # The real intervals would make this test a minute long.
                "BOOT_POLL_SECONDS": "0.2",
                "BOOT_RESTART_SECONDS": "0.2",
                "BOOT_STOP_SECONDS": "5",
            }
        )
        # POSIX path and an explicit cwd: on Windows this is git bash, which
        # does not take a backslash path as a script name.
        loop = subprocess.Popen(
            [bash, start.as_posix()],
            cwd=str(ws),
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            check("the supervisor starts the app it finds", _wait_for(ws / "ran", "v1"))
            first_pid = _pid(ws)

            # v2 arrives while v1 is running, and nobody signals anything: this
            # is the upload the receiver acknowledged and could not deliver.
            tarball = io.BytesIO()
            with tarfile.open(fileobj=tarball, mode="w:gz") as tar:
                body = fake_server.format(tag="v2").encode()
                info = tarfile.TarInfo("bootstrap.sh")
                info.size, info.mode = len(body), 0o755
                tar.addfile(info, io.BytesIO(body))
            # Atomically, as the receiver does: the supervisor polls for the
            # file's existence, so a half-written one would be claimed.
            (ws / "app.part").write_bytes(tarball.getvalue())
            os.replace(ws / "app.part", ws / "app.tgz")

            check(
                "an upload with no signal still replaces the running server",
                _wait_for(ws / "ran", "v2", timeout=30),
                _read(ws / "boot.log")[-800:],
            )
            check(
                "the upload was claimed",
                (ws / "app.installed.tgz").exists(),
                _read(ws / "boot.log")[-800:],
            )
            check("nothing is left pending", not (ws / "app.tgz").exists())
            check(
                "the old server was stopped, not left beside the new one",
                first_pid is not None and not _alive(first_pid),
                "pid %s is still running" % first_pid,
            )
        finally:
            loop.terminate()
            try:
                loop.wait(timeout=10)
            except Exception:  # pragma: no cover
                loop.kill()


def _bash():
    """A bash that can actually run a script.

    On Windows the `bash` on PATH is usually WSL's, which cannot execute a
    script at a Windows path (execvpe(/bin/bash) fails); git's is the one to
    use. On Linux, where this test matters most, PATH is right.
    """
    import subprocess

    candidates = []
    if os.name == "nt":
        candidates += [
            r"C:\Program Files\Git\bin\bash.exe",
            r"C:\Program Files\Git\usr\bin\bash.exe",
            os.path.expandvars(r"%LOCALAPPDATA%\Programs\Git\bin\bash.exe"),
        ]
    candidates.append(shutil.which("bash"))
    for candidate in candidates:
        if not candidate or not Path(candidate).exists():
            continue
        try:
            done = subprocess.run([candidate, "-c", "echo ok"], capture_output=True, timeout=30)
        except Exception:  # pragma: no cover - a broken interpreter
            continue
        if done.stdout.strip() == b"ok":
            return candidate
    return None


def _bash_path(path: Path) -> str:
    """A path the shell can use. On Windows that is git bash's /c/... form."""
    text = path.as_posix()
    if os.name == "nt" and len(text) > 2 and text[1] == ":":
        return "/" + text[0].lower() + text[2:]
    return text


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _wait_for(path: Path, needle: str, timeout: float = 20) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if needle in _read(path):
            return True
        time.sleep(0.2)
    return False


def _pid(ws: Path):
    try:
        return int(_read(ws / "server.pid").strip())
    except ValueError:
        return None


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


if __name__ == "__main__":
    test_tarball()
    test_start_cmd()
    test_receiver()
    test_user_agent()
    test_phase_contract()
    test_install_pending()
    test_watch_boot()
    test_idle_watch()
    test_watch_ctrl_c_leaves_the_pod_alone()
    test_watch_stops_rather_than_terminates()
    test_supervisor_swaps_a_running_server()
    print()
    if failures:
        print(f"{len(failures)} failed: {', '.join(failures)}")
        raise SystemExit(1)
    print("all checks passed")
