# swapctl

swapctl is a small self-hosted web UI that manages the full lifecycle of GGUF models for
[llama-swap](https://github.com/mostlygeek/llama-swap). It replaces the manual workflow of
`hf download + edit yaml + systemctl restart` with a browser interface: browse Hugging Face,
pick a quant, download it, auto-register it in `/etc/llama-swap/config.yaml`, and manage
removals and updates — all without touching the command line.

## Architecture

```
Browser
  |
  v
swapctl :8090
  |
  +-- HF API (search, file list, download)
  +-- /etc/llama-swap/config.yaml  (ruamel round-trip edit)
  +-- systemctl restart llama-swap
  +-- /models/<model-id>/          (GGUF files on disk)
  |
  v
llama-swap :8080
```

## Quickstart (deploy from dev machine)

```bash
./deploy.sh
```

The script rsyncs the source to `llm-gpu:/opt/swapctl`, installs dependencies in a venv,
installs the systemd unit, enables and starts the service, then verifies the API responds.

Override the SSH target with `--host`:

```bash
./deploy.sh --host my-gpu-box
```

## Local development

```bash
uv venv && source .venv/bin/activate
pip install -e '.[dev]'
uvicorn app.main:app --reload
```

The app is then available at `http://localhost:8090`.

## Configuration

swapctl is configured entirely through environment variables (or `/etc/swapctl.env`):

| Variable | Default | Description |
|---|---|---|
| `SWAPCTL_MODELS_DIR` | `/models` | Directory where GGUF files are stored |
| `SWAPCTL_CONFIG_PATH` | `/etc/llama-swap/config.yaml` | llama-swap config file |
| `SWAPCTL_REGISTRY_PATH` | `/var/lib/swapctl/registry.json` | swapctl state file |
| `SWAPCTL_SERVICE` | `llama-swap` | systemd service name to restart |
| `SWAPCTL_ENDPOINT` | `http://127.0.0.1:8080` | llama-swap API base URL |
| `SWAPCTL_PORT_BASE` | `9100` | Starting port for new llama-swap model blocks |
| `SWAPCTL_CTX_DEFAULT` | `65536` | Default context window size |
| `SWAPCTL_BIND_HOST` | `0.0.0.0` | Host uvicorn binds to |
| `SWAPCTL_BIND_PORT` | `8090` | Port uvicorn binds to |

## Security model

swapctl runs **as root** inside its LXC (it edits `/etc/llama-swap/config.yaml` and calls
`systemctl restart llama-swap`) and has **no authentication**. Mutating routes validate
`model_id` / `repo_id` / `filename` with strict regexes and `Path.resolve()` / `is_relative_to`
checks, so a malicious URL cannot escape `/models/` or inject llama-server flags — that is
the first line of defence. The second line is the network: swapctl must stay LAN-only.

Before exposing it beyond the LAN, add at least one of:

- Reverse proxy with HTTP basic auth or OIDC (nginx, Caddy, Tailscale Funnel)
- Bind it to `127.0.0.1` via `SWAPCTL_BIND_HOST=127.0.0.1` and tunnel via SSH / WireGuard
- A CSRF-aware middleware (swapctl does not set one; simple cross-origin POSTs from a
  malicious page on the same LAN would otherwise succeed)

Known limitations (documented, not yet fixed):
- Service runs as root; a future hardening pass should move `systemctl restart llama-swap`
  behind a narrow sudoers rule and drop to an unprivileged user
- No per-IP rate limiting on install/update endpoints
- CDN-hosted Tailwind/HTMX/Alpine.js without SRI hashes

## Troubleshooting

**View service logs:**

```bash
journalctl -u swapctl -f
```

**Reset the registry** (does not touch config.yaml or model files):

```bash
rm /var/lib/swapctl/registry.json
systemctl restart swapctl
```

**Inspect the llama-swap config manually:**

```bash
cat /etc/llama-swap/config.yaml
systemctl status llama-swap
curl -s http://127.0.0.1:8080/v1/models | python3 -m json.tool
```

**Service fails to start:**

Check that `/opt/swapctl/.venv/bin/python` exists. If not, re-run `./deploy.sh`.
Ensure `/var/lib/swapctl/` exists and is writable by root:

```bash
ls -ld /var/lib/swapctl/
```
