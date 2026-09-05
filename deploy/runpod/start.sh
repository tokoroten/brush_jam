#!/usr/bin/env bash
# The pod's dockerStartCmd, as a template. deploy.py substitutes the marker
# below with deploy/runpod/receiver.py verbatim and sends the result as
# ["bash", "-lc", "<this script>"] - there is no image to build and no
# registry to push to, so the whole bootstrap has to arrive this way.
#
# Responsibilities, in order: publish the receiver so the pod can be talked to
# at all, then wait for a tarball, extract it, and run its bootstrap.sh in a
# loop so that a re-upload (which pkills the server) restarts the new code.
set -u

mkdir -p /workspace
touch /workspace/boot.log
echo "waiting-for-upload" > /workspace/phase

cat > /workspace/receiver.py <<'RECEIVER_PY_EOF'
__RECEIVER_PY__
RECEIVER_PY_EOF

python3 /workspace/receiver.py >> /workspace/receiver.log 2>&1 &

while true; do
  if [ ! -f /workspace/app.tgz ]; then
    sleep 3
    continue
  fi
  if [ ! -f /workspace/.stamp ] || [ /workspace/app.tgz -nt /workspace/.stamp ]; then
    echo "extract" > /workspace/phase
    echo "[start] extracting $(stat -c %s /workspace/app.tgz) bytes" >> /workspace/boot.log
    rm -rf /workspace/app.new
    mkdir -p /workspace/app.new
    if tar xzf /workspace/app.tgz -C /workspace/app.new; then
      rm -rf /workspace/app.old
      if [ -d /workspace/app ]; then mv /workspace/app /workspace/app.old; fi
      mv /workspace/app.new /workspace/app
      touch /workspace/.stamp
    else
      echo "[start] tarball did not extract; waiting for another upload" >> /workspace/boot.log
      rm -f /workspace/app.tgz
      echo "waiting-for-upload" > /workspace/phase
      sleep 3
      continue
    fi
  fi
  bash /workspace/app/bootstrap.sh >> /workspace/boot.log 2>&1
  echo "[start] server exited ($?); restarting in 3s" >> /workspace/boot.log
  echo "restarting" > /workspace/phase
  sleep 3
done
