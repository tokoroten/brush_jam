#!/usr/bin/env python3
"""Put Brush Jam on a RunPod GPU Pod, without SSH, a git remote or a registry.

    uv run --project apps/brushjam python deploy/runpod/deploy.py deploy
    ...                                                          status
    ...                                                          log
    ...                                                          upload
    ...                                                          stop | start | terminate
    ...                                                          watch --idle-minutes 30

How it works: the pod runs a stock PyTorch image whose dockerStartCmd is
`deploy/runpod/start.sh` with `receiver.py` inlined. The receiver listens on
:8788 (proxied by RunPod over HTTPS) and is the only way in; this script PUTs a
tarball of `apps/brushjam` to it, and the pod's boot loop extracts that and runs
`bootstrap.sh`. Uploading again replaces the code and restarts the server.

Secrets (RUNPOD_API_KEY, HF_TOKEN, CIVITAI_TOKEN) come from the repo-root .env
and are never printed - not by this script, not into boot.log, not into docs.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import secrets
import sys
import tarfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
POD_FILE = HERE / ".pod"
TARBALL = HERE / "app.tgz"
API = "https://rest.runpod.io/v1"
#: Anything but urllib's default: the pod proxy 403s that one.
USER_AGENT = "brushjam-deploy/1.0"

GPU_TYPE = "NVIDIA GeForce RTX 4090"
IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
CHECKPOINT = "/workspace/models/checkpoints/sdxl-checkpoint.safetensors"
CIVITAI_VERSION = "2940478"


# --------------------------------------------------------------------------
# secrets and small helpers


def load_env() -> Dict[str, str]:
    """Repo-root .env, via python-dotenv when it is installed."""
    path = REPO / ".env"
    try:
        from dotenv import dotenv_values

        values = {k: v for k, v in dotenv_values(path).items() if v is not None}
    except ImportError:
        values = {}
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                values[k.strip()] = v.strip().strip('"').strip("'")
    for key in ("RUNPOD_API_KEY", "HF_TOKEN", "CIVITAI_TOKEN"):
        if os.environ.get(key):
            values[key] = os.environ[key]
    return values


def require(env: Dict[str, str], key: str) -> str:
    value = env.get(key)
    if not value:
        sys.exit(f"{key} is not set (repo-root .env)")
    return value


def api(env: Dict[str, str], method: str, path: str, body: Optional[dict] = None) -> Any:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        f"{API}{path}",
        data=data,
        method=method,
        headers={
            "authorization": f"Bearer {require(env, 'RUNPOD_API_KEY')}",
            "content-type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as response:
            raw = response.read()
    except urllib.error.HTTPError as err:  # the body says what was wrong
        detail = err.read().decode("utf-8", "replace")[:2000]
        sys.exit(f"RunPod {method} {path} -> {err.code}: {detail}")
    return json.loads(raw) if raw else None


def read_pod() -> Dict[str, str]:
    if not POD_FILE.exists():
        sys.exit(f"no pod recorded; run `deploy` first ({POD_FILE} is missing)")
    return json.loads(POD_FILE.read_text(encoding="utf-8"))


def proxy(pod_id: str, port: int) -> str:
    return f"https://{pod_id}-{port}.proxy.runpod.net"


def proxy_request(url: str, data: Optional[bytes] = None, method: str = "GET", **headers: str):
    """A request the RunPod proxy will actually forward.

    The proxy answers 403 to anything sent with urllib's default user agent -
    the same request with curl's goes through - so every call through
    *.proxy.runpod.net names itself. This looked exactly like a bad upload
    token for two rounds of debugging.
    """
    request = urllib.request.Request(url, data=data, method=method)
    request.add_header("user-agent", USER_AGENT)
    for name, value in headers.items():
        request.add_header(name.replace("_", "-"), value)
    return request


def get(url: str, timeout: float = 20) -> str:
    with urllib.request.urlopen(proxy_request(url), timeout=timeout) as response:
        return response.read().decode("utf-8", "replace")


# --------------------------------------------------------------------------
# the tarball


def build_tarball(dest: Path = TARBALL) -> Path:
    """apps/brushjam plus bootstrap.sh, and nothing else.

    The static web build has to be in the tree already (`pnpm build`); the
    pod has no Node, so a missing one would only surface as a 404 in a
    friend's browser.
    """
    app = REPO / "apps" / "brushjam"
    if not (app / "src" / "brushjam" / "static" / "index.html").exists():
        sys.exit("apps/brushjam/src/brushjam/static/index.html is missing; run `pnpm build` first")

    skip_dirs = {"__pycache__", ".venv", ".pytest_cache", ".ruff_cache", "node_modules", ".git"}

    def keep(info: tarfile.TarInfo) -> Optional[tarfile.TarInfo]:
        parts = Path(info.name).parts
        if any(part in skip_dirs for part in parts):
            return None
        if info.name.endswith((".pyc", ".log")):
            return None
        info.uid = info.gid = 0
        info.uname = info.gname = "root"
        info.mode = 0o755 if info.isdir() or info.name.endswith(".sh") else 0o644
        return info

    # bootstrap.sh is edited on Windows; a CRLF in it is `$'\r': command not
    # found` in the pod, so it goes in as bytes with the line endings fixed.
    script = (HERE / "bootstrap.sh").read_bytes().replace(b"\r\n", b"\n")

    tmp = dest.with_suffix(".part")
    with tarfile.open(tmp, "w:gz") as tar:
        tar.add(app, arcname="apps/brushjam", filter=keep)
        info = tarfile.TarInfo("bootstrap.sh")
        info.size, info.mode, info.mtime = len(script), 0o755, int(time.time())
        tar.addfile(info, io.BytesIO(script))
    tmp.replace(dest)
    return dest


def docker_start_cmd() -> list:
    """start.sh with receiver.py inlined, as the pod's command array."""
    script = (HERE / "start.sh").read_text(encoding="utf-8")
    receiver = (HERE / "receiver.py").read_text(encoding="utf-8")
    if "RECEIVER_PY_EOF" in receiver:
        raise ValueError("receiver.py contains the heredoc terminator")
    return ["bash", "-lc", script.replace("__RECEIVER_PY__", receiver.rstrip("\n"))]


def pod_env(env: Dict[str, str], upload_token: str) -> Dict[str, str]:
    return {
        "UPLOAD_TOKEN": upload_token,
        "HF_TOKEN": require(env, "HF_TOKEN"),
        "CIVITAI_TOKEN": require(env, "CIVITAI_TOKEN"),
        "CIVITAI_VERSION": CIVITAI_VERSION,
        "HF_HOME": "/workspace/hf",
        "INPROC_CHECKPOINT": CHECKPOINT,
        "INPROC_LORA_DIR": "/workspace/models/loras",
        "AI_BACKEND": "inproc",
        "HOST": "0.0.0.0",
        "PORT": "8787",
        # Every player reaches the server through the RunPod proxy, so they all
        # share one client IP and the per-IP room-creation limit applies to the
        # whole group.
        "ROOM_CREATE_PER_MIN": "60",
    }


# --------------------------------------------------------------------------
# subcommands


def cmd_build(_args: argparse.Namespace) -> None:
    path = build_tarball()
    print(f"{path} ({path.stat().st_size / 1e6:.1f} MB)")


def cmd_deploy(args: argparse.Namespace) -> None:
    env = load_env()
    if POD_FILE.exists() and not args.force:
        sys.exit(f"{POD_FILE} already exists; use `upload` for new code, or --force for a second pod")
    tarball = build_tarball()
    token = secrets.token_urlsafe(24)
    print(f"creating a pod ({GPU_TYPE}, SECURE)", flush=True)
    pod = api(
        env,
        "POST",
        "/pods",
        {
            "name": args.name,
            "imageName": IMAGE,
            "gpuTypeIds": [GPU_TYPE],
            "gpuCount": 1,
            "cloudType": "SECURE",
            "computeType": "GPU",
            "volumeInGb": 40,
            "volumeMountPath": "/workspace",
            "containerDiskInGb": 20,
            "ports": ["8787/http", "8788/http"],
            "env": pod_env(env, token),
            "dockerStartCmd": docker_start_cmd(),
            # Hosts differ wildly in bandwidth; a slow one turns the 10 GB
            # first boot into an hour. Ask for a fast one and enough vCPUs.
            "minDownloadMbps": args.min_download,
            "minVCPUPerGPU": args.min_vcpu,
            **({"dataCenterIds": args.data_center} if args.data_center else {}),
        },
    )
    pod_id = pod["id"]
    POD_FILE.write_text(json.dumps({"id": pod_id, "token": token}, indent=2), encoding="utf-8")
    print(f"pod {pod_id} created ({pod.get('costPerHr', '?')} $/hr); .pod written", flush=True)

    print("waiting for the receiver to answer (a cold machine takes a few minutes)", flush=True)
    deadline = time.time() + args.wait
    while time.time() < deadline:
        try:
            get(f"{proxy(pod_id, 8788)}/status", timeout=10)
            break
        except Exception:
            time.sleep(5)
    else:
        sys.exit("the receiver never answered; check `status` and the RunPod console")

    upload(pod_id, token, tarball)
    print(f"\nplay at {proxy(pod_id, 8787)}/")
    print("first boot downloads ~7 GB of models; follow it with `log` / `status`")


def upload(pod_id: str, token: str, tarball: Path) -> None:
    # Bytes, not the file object: urllib sends a file body chunked, and a
    # chunked PUT never reaches the receiver with a length it can store (the
    # proxy refuses it outright). The tarball is under a megabyte.
    body = tarball.read_bytes()
    print(f"uploading {len(body) / 1e6:.1f} MB", flush=True)
    req = proxy_request(
        f"{proxy(pod_id, 8788)}/upload?token={token}",
        data=body,
        method="PUT",
        content_type="application/octet-stream",
        content_length=str(len(body)),
    )
    with urllib.request.urlopen(req, timeout=600) as response:
        print(f"receiver: {response.read().decode().strip()}", flush=True)


def watch_boot(pod_id: str, timeout: float) -> bool:
    """Print each phase change until the server answers /healthz."""
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        try:
            status = json.loads(get(proxy(pod_id, 8788) + "/status", timeout=15))
        except Exception as err:
            if last != f"unreachable: {err}":
                last = f"unreachable: {err}"
                print(f"  {last}", flush=True)
            time.sleep(5)
            continue
        if status.get("server_healthy"):
            print("  server_healthy", flush=True)
            return True
        phase = f"{status.get('phase')} (app={status.get('app_present')}, checkpoint={status.get('checkpoint_present')})"
        if phase != last:
            print(f"  {phase}", flush=True)
            last = phase
        time.sleep(10)
    return False


def cmd_upload(_args: argparse.Namespace) -> None:
    pod = read_pod()
    upload(pod["id"], pod["token"], build_tarball())


def cmd_status(_args: argparse.Namespace) -> None:
    pod = read_pod()
    remote = api(load_env(), "GET", f"/pods/{pod['id']}")
    print(f"pod        {pod['id']}  {remote.get('desiredStatus')}  {remote.get('costPerHr', '?')} $/hr")
    machine = remote.get("machine") or {}
    if machine.get("gpuTypeId"):
        print(f"gpu        {machine['gpuTypeId']}")
    try:
        print(f"boot       {get(proxy(pod['id'], 8788) + '/status')}")
    except Exception as err:
        print(f"boot       receiver unreachable ({err})")
    try:
        print(f"server     {get(proxy(pod['id'], 8787) + '/healthz')}")
    except Exception as err:
        print(f"server     not serving yet ({err})")
    print(f"url        {proxy(pod['id'], 8787)}/")


def cmd_log(args: argparse.Namespace) -> None:
    pod = read_pod()
    url = proxy(pod["id"], 8788) + "/log"
    if not args.follow:
        print(get(url, timeout=30))
        return
    seen = ""
    while True:
        try:
            text = get(url, timeout=30)
        except Exception as err:
            print(f"[log] {err}")
            time.sleep(5)
            continue
        if text.startswith(seen):
            sys.stdout.write(text[len(seen) :])
        else:
            sys.stdout.write(text)
        sys.stdout.flush()
        seen = text
        time.sleep(3)


def cmd_stop(_args: argparse.Namespace) -> None:
    pod = read_pod()
    api(load_env(), "POST", f"/pods/{pod['id']}/stop")
    print(f"pod {pod['id']} stopping; /workspace survives, `start` brings it back")


def cmd_start(_args: argparse.Namespace) -> None:
    pod = read_pod()
    api(load_env(), "POST", f"/pods/{pod['id']}/start")
    print(f"pod {pod['id']} starting; the tarball and models are still on /workspace")


# --------------------------------------------------------------------------
# the idle watcher


#: How often `watch` asks the pod how it is doing.
WATCH_INTERVAL_SECONDS = 60


class IdleTracker:
    """Is anybody using the pod?

    Two signals, both from the server's own /healthz: `active_sockets` (rooms
    with somebody in them) and `last_generation_at` (when the model last
    produced anything). A room sitting open with nobody drawing is still
    somebody's room, so sockets alone keep the pod alive.

    A server too old to report either is `unknown`, and the pod is never
    stopped on it: a watcher that cannot see the players must not turn the
    lights off.

    The deadline is the later of those two signals plus the idle window - so a
    watch started beside a pod that has been idle for hours stops it at the
    first poll, rather than granting it another window for having been idle.
    """

    def __init__(self, idle_seconds: float) -> None:
        self.idle_seconds = float(idle_seconds)
        self.state: Optional[str] = None
        self.detail = ""
        #: When the pod was last *used*, on the caller's clock - not when it
        #: was first seen unused. The pod is stopped `idle_seconds` after this,
        #: which is the whole rule.
        #:
        #: Counting from the first idle poll instead charged two windows for
        #: one generation: the generation kept the pod "busy" for the entire
        #: window, and only then did the idle clock start, so a default watch
        #: stopped an hour after the last picture rather than half an hour.
        self.active_at: Optional[float] = None
        #: When /healthz first stopped answering, cleared by every reply. An
        #: outage is its own window: a single failed request seconds after a
        #: healthy poll is a hiccup, not thirty minutes of silence, and it used
        #: to stop the pod outright by inheriting the last-use deadline.
        self.unreachable_since: Optional[float] = None
        self._generation_marker: Any = None

    def observe(self, health: Optional[Dict[str, Any]], at: float) -> bool:
        """Take one poll. True if the situation changed and is worth printing."""
        state, detail, activity = self._classify(health, at)
        if state == "unreachable":
            if self.unreachable_since is None:
                self.unreachable_since = at
        else:
            self.unreachable_since = None
        if state == "unknown":
            self.active_at = None
        elif activity is not None:
            self.active_at = activity if self.active_at is None else max(self.active_at, activity)
        elif self.active_at is None:
            # Nothing has happened yet and nothing says when anything last did:
            # the window starts when the watch does.
            self.active_at = at
        changed = (state, detail) != (self.state, self.detail)
        self.state, self.detail = state, detail
        return changed

    def _classify(self, health: Optional[Dict[str, Any]], at: float):
        """(state, detail, when the pod was last used) for one poll."""
        if not isinstance(health, dict):
            return "unreachable", "", None
        sockets = health.get("active_sockets")
        generation = health.get("last_generation_at")
        marker = (generation, health.get("generations"))
        moved = self._generation_marker is not None and marker != self._generation_marker
        self._generation_marker = marker
        if sockets is None and generation is None:
            return "unknown", "this server does not report activity; it will not be stopped", None
        if isinstance(sockets, int) and sockets > 0:
            return (
                "busy",
                "%d socket%s connected" % (sockets, "" if sockets == 1 else "s"),
                at,
            )
        if moved:
            return "busy", "a generation since the last poll", at
        if isinstance(generation, (int, float)):
            # Epoch milliseconds from the pod, against our own clock. The
            # generation happened then, not now: it is activity at its own
            # time, and the deadline is that time plus the window.
            since = at - generation / 1000.0
            if since >= 0:
                if since < self.idle_seconds:
                    return "busy", "a generation %d s ago" % round(since), at - since
                return "idle", "nobody connected", at - since
        return "idle", "nobody connected", None

    def idle_for(self, at: float) -> float:
        return 0.0 if self.active_at is None else max(0.0, at - self.active_at)

    def describe(self, at: float) -> str:
        if self.state == "busy":
            return "busy: %s" % self.detail
        if self.state == "unknown":
            return "unknown: %s" % self.detail
        if self.state == "unreachable":
            out = 0.0 if self.unreachable_since is None else max(0.0, at - self.unreachable_since)
            return "unreachable (%d/%d min)" % (int(out // 60), int(self.idle_seconds // 60))
        return "idle (%d/%d min)" % (int(self.idle_for(at) // 60), int(self.idle_seconds // 60))

    def stop_reason(self, at: float) -> Optional[str]:
        """Why the pod should be stopped now, or None to keep watching."""
        minutes = int(self.idle_seconds // 60)
        if self.state == "unreachable":
            # The outage is timed from its own start. Otherwise one failed
            # request just after a generation 29 minutes old stopped the pod
            # and announced half an hour of silence that never happened.
            if self.unreachable_since is None or at - self.unreachable_since < self.idle_seconds:
                return None
            return "the server has not answered /healthz for %d minutes" % minutes
        if self.state != "idle":
            return None
        if self.active_at is None or self.idle_for(at) < self.idle_seconds:
            return None
        return "nobody has been connected and nothing has been generated for %d minutes" % minutes


def watch_loop(
    poll,
    *,
    idle_seconds: float,
    interval: float,
    now,
    sleep,
    log=print,
) -> str:
    """Poll until the pod has been idle long enough. Returns why it stopped.

    Everything it touches is injected, so the whole decision - including the
    clock - is exercised by test_deploy.py without a pod or a minute of
    waiting.
    """
    tracker = IdleTracker(idle_seconds)
    while True:
        at = now()
        if tracker.observe(poll(), at):
            log("  %s" % tracker.describe(at))
        reason = tracker.stop_reason(at)
        if reason:
            return reason
        sleep(interval)


def health_poller(pod_id: str):
    """/healthz as a dict, or None when the pod does not answer."""

    def poll() -> Optional[Dict[str, Any]]:
        try:
            return json.loads(get(proxy(pod_id, 8787) + "/healthz", timeout=20))
        except Exception:
            return None

    return poll


def cmd_watch(args: argparse.Namespace) -> None:
    pod = read_pod()
    what = "terminate" if args.terminate else "stop"
    print(
        "watching %s every %ds; will %s it after %g idle minutes (Ctrl+C leaves it alone)"
        % (pod["id"], int(args.interval), what, args.idle_minutes),
        flush=True,
    )
    try:
        reason = watch_loop(
            health_poller(pod["id"]),
            idle_seconds=args.idle_minutes * 60,
            interval=args.interval,
            now=time.time,
            sleep=time.sleep,
        )
    except KeyboardInterrupt:
        # Deliberately not a stop: the pod is left exactly as it was found.
        print("\nwatch ended; the pod is untouched")
        return
    print("%s: %s" % (what.replace("stop", "stopping").replace("terminate", "terminating"), reason))
    env = load_env()
    if args.terminate:
        api(env, "DELETE", "/pods/%s" % pod["id"])
        POD_FILE.unlink(missing_ok=True)
        print("pod %s terminated; .pod removed" % pod["id"])
        return
    api(env, "POST", "/pods/%s/stop" % pod["id"])
    print("pod %s stopping; /workspace survives, `start` brings it back" % pod["id"])


def cmd_terminate(args: argparse.Namespace) -> None:
    pod = read_pod()
    if not args.yes:
        sys.exit("terminate destroys the volume and the downloaded models; pass --yes")
    api(load_env(), "DELETE", f"/pods/{pod['id']}")
    POD_FILE.unlink(missing_ok=True)
    print(f"pod {pod['id']} terminated; .pod removed")


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("deploy", help="create the pod and upload the code")
    p.add_argument("--name", default="brushjam")
    p.add_argument("--wait", type=float, default=900, help="seconds to wait for the receiver")
    p.add_argument("--boot-wait", type=float, default=1800, help="seconds to wait for the first model load")
    p.add_argument("--force", action="store_true", help="deploy even though .pod exists")
    p.add_argument("--min-download", type=int, default=800, help="minimum host download Mbps (default 800)")
    p.add_argument("--min-vcpu", type=int, default=4, help="minimum vCPUs per GPU (default 4)")
    p.add_argument("--data-center", action="append", help="restrict to a data center id (repeatable), e.g. EU-RO-1")
    p.set_defaults(func=cmd_deploy)

    p = sub.add_parser("build", help="build the tarball only")
    p.set_defaults(func=cmd_build)

    p = sub.add_parser("upload", help="rebuild the tarball and restart the server with it")
    p.set_defaults(func=cmd_upload)

    p = sub.add_parser("status", help="pod state, boot phase and server health")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("log", help="the last 200 lines of /workspace/boot.log")
    p.add_argument("-f", "--follow", action="store_true")
    p.set_defaults(func=cmd_log)

    sub.add_parser("stop", help="stop the pod (keeps the volume)").set_defaults(func=cmd_stop)
    sub.add_parser("start", help="start a stopped pod").set_defaults(func=cmd_start)

    p = sub.add_parser("watch", help="stop the pod once nobody is using it")
    p.add_argument(
        "--idle-minutes",
        type=float,
        default=30,
        help="stop after this long with no sockets and no generations (default 30)",
    )
    p.add_argument(
        "--interval",
        type=float,
        default=WATCH_INTERVAL_SECONDS,
        help="seconds between /healthz polls (default 60)",
    )
    p.add_argument(
        "--terminate",
        action="store_true",
        help="destroy the pod and its volume instead of stopping it",
    )
    p.set_defaults(func=cmd_watch)

    p = sub.add_parser("terminate", help="destroy the pod and its volume")
    p.add_argument("--yes", action="store_true")
    p.set_defaults(func=cmd_terminate)

    # Piped output is block-buffered, which makes a 15-minute deploy look hung.
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except AttributeError:
        pass

    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
