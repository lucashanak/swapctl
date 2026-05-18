#!/usr/bin/env bash
set -euo pipefail

SSH_TARGET="llm-gpu"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --host)
            SSH_TARGET="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Syncing to ${SSH_TARGET}:/opt/swapctl/ ..."
rsync -av --delete \
    --exclude '__pycache__' \
    --exclude '.venv' \
    --exclude '*.pyc' \
    --exclude '.git' \
    --exclude '.omc' \
    --exclude '.pytest_cache' \
    "${SCRIPT_DIR}/" \
    "${SSH_TARGET}:/opt/swapctl/"

echo "Running remote setup on ${SSH_TARGET} ..."
ssh "${SSH_TARGET}" bash <<'REMOTE'
set -euo pipefail

if [ ! -d /opt/swapctl/.venv ]; then
    if ! python3 -c 'import ensurepip' 2>/dev/null; then
        echo "Installing python3-venv ..."
        DEBIAN_FRONTEND=noninteractive apt-get install -y python3-venv python3-pip
    fi
    echo "Creating virtualenv ..."
    python3 -m venv /opt/swapctl/.venv
fi

echo "Installing dependencies ..."
/opt/swapctl/.venv/bin/pip install -e /opt/swapctl/ --quiet

echo "Creating /var/lib/swapctl/ ..."
mkdir -p /var/lib/swapctl
chmod 700 /var/lib/swapctl

echo "Installing systemd unit ..."
cp /opt/swapctl/systemd/swapctl.service /etc/systemd/system/swapctl.service
systemctl daemon-reload

echo "Enabling and starting swapctl ..."
if systemctl is-active --quiet swapctl; then
    systemctl restart swapctl
else
    systemctl enable --now swapctl
fi
REMOTE

echo "Waiting 5s for service to come up ..."
sleep 5

echo "Checking API status ..."
curl -sf "http://192.168.1.94:8090/api/status" && echo ""
echo "Deploy complete."
