#!/usr/bin/env bash
# The pod's dockerStartCmd, as a template. deploy.py substitutes the marker
# below with deploy/runpod/receiver.py verbatim and sends the result as
# ["bash", "-lc", "<this script>"] - there is no image to build and no
# registry to push to, so the whole bootstrap has to arrive this way.
#
# Responsibilities, in order: publish the receiver so the pod can be talked to
# at all, then wait for a tarball, install it, and run its bootstrap.sh in a
# loop so that a re-upload (which stops the server) restarts the new code.
set -u

WS="${WORKSPACE_DIR:-/workspace}"
mkdir -p "$WS"
touch "$WS/boot.log"

# An upload is "pending" precisely while /workspace/app.tgz exists, and it is
# claimed by renaming it away - one atomic step, no timestamps. Comparing the
# tarball's mtime against a stamp written *after* extraction lost an upload for
# good: a tarball that arrived during the previous extraction was acknowledged
# by the receiver, then looked older than the stamp and was never installed.
install_pending() {
  [ -f "$WS/app.tgz" ] || return 1
  local claim="$WS/app.claimed.tgz"
  rm -f "$claim"
  mv "$WS/app.tgz" "$claim" || return 1
  echo "extract" > "$WS/phase"
  echo "[start] extracting $(stat -c %s "$claim") bytes" >> "$WS/boot.log"
  rm -rf "$WS/app.new"
  mkdir -p "$WS/app.new"
  if ! tar xzf "$claim" -C "$WS/app.new"; then
    echo "[start] that tarball did not extract; waiting for another upload" >> "$WS/boot.log"
    rm -rf "$WS/app.new"
    rm -f "$claim"
    echo "waiting-for-upload" > "$WS/phase"
    return 1
  fi
  rm -rf "$WS/app.old"
  if [ -d "$WS/app" ]; then mv "$WS/app" "$WS/app.old"; fi
  mv "$WS/app.new" "$WS/app"
  mv -f "$claim" "$WS/app.installed.tgz"
  echo "[start] installed" >> "$WS/boot.log"
  return 0
}

# Sourced by deploy/runpod/test_deploy.py to exercise install_pending.
if [ -n "${START_SH_FUNCTIONS_ONLY:-}" ]; then
  return 0 2>/dev/null || exit 0
fi

# The server runs as a CHILD of this loop, and the loop watches for two things
# at once: the child exiting, and a tarball landing. Running bootstrap.sh in
# the foreground meant an upload that arrived between `install_pending` finding
# nothing and bootstrap.sh publishing its pid was acknowledged by the receiver,
# signalled at a stale pid that no longer existed, and then waited for forever:
# the old server ran on, the loop blocked on it, and the new code sat in
# /workspace/app.tgz unnoticed.
run_app() {
  bash "$WS/app/bootstrap.sh" >> "$WS/boot.log" 2>&1 &
  local child=$!
  local waited=0
  while kill -0 "$child" 2>/dev/null; do
    if [ -f "$WS/app.tgz" ]; then
      echo "[start] an upload is waiting; stopping pid $child" >> "$WS/boot.log"
      kill -TERM "$child" 2>/dev/null
      waited=0
      while kill -0 "$child" 2>/dev/null && [ "$waited" -lt "${BOOT_STOP_SECONDS:-30}" ]; do
        sleep 1
        waited=$((waited + 1))
      done
      kill -KILL "$child" 2>/dev/null
      break
    fi
    sleep "${BOOT_POLL_SECONDS:-2}"
  done
  # Reaped here, so the replacement never starts beside a server still holding
  # the GPU and port 8787.
  wait "$child" 2>/dev/null
  return $?
}

echo "waiting-for-upload" > "$WS/phase"

cat > "$WS/receiver.py" <<'RECEIVER_PY_EOF'
__RECEIVER_PY__
RECEIVER_PY_EOF

python3 "$WS/receiver.py" >> "$WS/receiver.log" 2>&1 &

while true; do
  # Always install a pending upload first, and loop back to look again: one
  # may have landed while the last one was being extracted, and starting the
  # server on the older code would strand it there.
  if install_pending; then
    continue
  fi
  if [ ! -d "$WS/app" ]; then
    sleep "${BOOT_RESTART_SECONDS:-3}"
    continue
  fi
  run_app
  rc=$?
  echo "[start] server exited ($rc)" >> "$WS/boot.log"
  echo "restarting" > "$WS/phase"
  # An upload is already waiting: install it now rather than idling first.
  [ -f "$WS/app.tgz" ] && continue
  sleep "${BOOT_RESTART_SECONDS:-3}"
done
