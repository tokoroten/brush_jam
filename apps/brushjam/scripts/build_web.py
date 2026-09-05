"""Build the web client and copy it into the package.

    uv run python scripts/build_web.py

Runs `pnpm --filter @brushjam/web build` and copies `apps/web/dist` into
`src/brushjam/static`, which is what `brushjam` serves when it is installed
somewhere without the repo around it. In a dev checkout the server falls back to
`apps/web/dist` directly, so this is only needed for a packaged run.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
APP = HERE.parent
REPO = APP.parent.parent
DIST = REPO / "apps" / "web" / "dist"
STATIC = APP / "src" / "brushjam" / "static"


def main() -> int:
    if "--no-build" not in sys.argv:
        pnpm = shutil.which("pnpm") or shutil.which("pnpm.cmd")
        if pnpm is None:
            print("pnpm is not on PATH; build apps/web yourself or pass --no-build")
            return 1
        result = subprocess.run([pnpm, "--filter", "@brushjam/web", "build"], cwd=REPO)
        if result.returncode != 0:
            return result.returncode
    if not (DIST / "index.html").exists():
        print(f"{DIST} has no index.html")
        return 1
    if STATIC.exists():
        shutil.rmtree(STATIC)
    shutil.copytree(DIST, STATIC)
    print(f"[build_web] copied {DIST} -> {STATIC}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
