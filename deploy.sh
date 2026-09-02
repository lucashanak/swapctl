#!/usr/bin/env bash
set -euo pipefail

SSH_TARGET="llm-gpu"
FORCE=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --host)
            SSH_TARGET="$2"
            shift 2
            ;;
        --force)
            FORCE=1
            shift
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Deploying restarts swapctl, which silently destroys any running test job — the
# run keeps looking alive from the outside while its result can never arrive.
# That has cost four separate multi-hour runs. Refuse instead.
#
# The check runs ON the node: /api/status needs a session cookie, so an
# unauthenticated curl from here just returns {"detail": ...} and the guard
# would pass every time — which is exactly how the first version of it failed.
if [[ "${FORCE}" -eq 0 ]]; then
    ACTIVE="$(ssh "${SSH_TARGET}" 'python3 - <<PYEOF
import json, urllib.request, urllib.parse, urllib.error
BASE = "http://127.0.0.1:8090/manage"
pw = None
for ln in open("/etc/swapctl.env"):
    if ln.startswith("SWAPCTL_PASSWORD="):
        pw = ln.split("=", 1)[1].strip().strip(chr(34)).strip(chr(39))
class N(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *a, **k):
        return None
op = urllib.request.build_opener(N)
ck = None
try:
    op.open(BASE + "/login", data=urllib.parse.urlencode({"password": pw}).encode(), timeout=15)
except urllib.error.HTTPError as e:
    for c in (e.headers.get_all("Set-Cookie") or []):
        if c.startswith("swapctl_session="):
            ck = c.split(";")[0]
except Exception:
    raise SystemExit(0)
if not ck:
    raise SystemExit(0)
try:
    req = urllib.request.Request(BASE + "/api/status", headers={"Cookie": ck})
    d = json.loads(urllib.request.urlopen(req, timeout=15).read())
except Exception:
    raise SystemExit(0)
for j in d.get("active_tests") or []:
    print("%s (%s, %s)" % (j.get("model_id"), j.get("test_type"), j.get("status")))
PYEOF' 2>/dev/null || true)"
    if [[ -n "${ACTIVE}" ]]; then
        echo "REFUSING TO DEPLOY - a test is running and the restart would kill it:" >&2
        echo "${ACTIVE}" | sed 's/^/  /' >&2
        echo "" >&2
        echo "Wait for it to finish, or re-run with --force to deploy anyway." >&2
        exit 1
    fi
fi

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
