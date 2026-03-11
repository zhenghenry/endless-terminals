#!/usr/bin/env bash
set -euo pipefail

# Helper: run as root (Docker usually is), otherwise use sudo if present.
run() {
  if [[ "$(id -u)" -eq 0 ]]; then
    "$@"
  elif command -v sudo >/dev/null 2>&1; then
    sudo "$@"
  else
    echo "ERROR: Need root privileges, but 'sudo' is not available. Re-run as root (or install sudo)." >&2
    exit 1
  fi
}

# 1) Update & add the Apptainer PPA
run apt-get update
run apt-get install -y --no-install-recommends software-properties-common ca-certificates gnupg
run add-apt-repository -y ppa:apptainer/ppa
run apt-get update

# 2) Install
# Rootless by default:
# run apt-get install -y apptainer
# If you specifically want setuid mode:
run apt-get install -y apptainer-suid

# 3) Remount /proc (often not permitted in unprivileged containers)
if run mount -o remount,hidepid=0 /proc 2>/dev/null; then
  echo "Remounted /proc with hidepid=0"
else
  echo "WARN: Could not remount /proc (likely needs --privileged or extra capabilities). Skipping."
fi
