# swapctl

swapctl is a self-hosted web UI that manages the full lifecycle of GGUF models for
[llama-swap](https://github.com/mostlygeek/llama-swap). Browse Hugging Face, pick a
quant, download it, auto-register it in `/etc/llama-swap/config.yaml`, run IQ
benchmarks, manage TTS voices, and transcribe audio — all from the browser.

## Architecture

```
Browser
  |
  v
swapctl :8090  (FastAPI + HTMX + Alpine.js)
  |
  +-- HF API (search, file list, download)
  +-- /etc/llama-swap/config.yaml  (ruamel round-trip edit)
  +-- systemctl restart llama-swap
  +-- /models/<model-id>/          (GGUF files on disk)
  +-- /models/voices/              (Piper ONNX voice files)
  |
  v
llama-swap :8080
  +-- llama.cpp backends (LLM, Whisper STT)
```

## Quickstart (deploy from dev machine)

```bash
# Sync app/ only (safe when remote has no internet — skips pip)
rsync -a app/ pve2-llm:/opt/swapctl/app/
ssh pve2-llm 'systemctl restart swapctl'

# Full deploy (installs venv + deps — requires internet on remote)
./deploy.sh
```

> **Note:** Do not run `deploy.sh` if the remote host has no internet access. Use the
> `rsync app/` pattern instead — it only ships Python source, which needs no pip step.

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

## Features

### Model management
Browse Hugging Face, download GGUF files, auto-register in llama-swap config, run
removal and updates, view VRAM estimates.

### IQ benchmark
Automated LLM intelligence testing with configurable categories (math, reasoning,
code, Czech, …). Thinking mode is auto-detected per model (no manual flag needed).
Results include per-question token counts and total duration for effectivity ranking
(IQ × 0.7 + speed × 0.3).

### Speed benchmark
Measures generation tokens/sec and time-to-first-token per model. History is
preserved across restarts.

### Backend registry / per-model backend switching
Each model can be assigned a specific llama-swap backend (e.g. `llama-cpp-vulkan`,
`llama-cpp-rocm`, `whisper-server`). The registry performs an architecture
compatibility check before writing the config — mismatched backends are rejected with
a clear error. Speculative decoding draft pairings are configured separately per
model in the UI.

### Audio — TTS (Piper)
`POST /api/v1/audio/speech` synthesizes text to WAV. Voices are loaded in-process via
the `piper-tts` ONNX package on first use (~3 s) and cached in RAM for subsequent
calls (~0.3 s). RAM cost: ~150 MB per medium-quality voice. Manage voices at
`/manage/voices` or via the install/uninstall API endpoints.

### Audio — STT (Whisper)
`POST /api/v1/audio/transcriptions` transcribes audio via a llama-swap
whisper-server backend. Compatible with the OpenAI `/v1/audio/transcriptions` shape.

### Voice manager page
`/manage/voices` lists installed voices, shows the full Piper catalog, and lets you
install or remove voices with one click. Includes an in-browser preview player.

### API reference page
`/manage/api-docs` documents all public and management endpoints with copy-pastable
`curl` examples. See it for full endpoint details.

## Configuration

swapctl is configured through environment variables (or `/etc/swapctl.env`):

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

swapctl runs **as root** inside its LXC (it edits `/etc/llama-swap/config.yaml` and
calls `systemctl restart llama-swap`). The UI requires a session cookie (login page at
`/manage/login`). Audio API endpoints (`/api/v1/audio/*`) are public — no auth
required, by design, for OpenWebUI and external integrator compatibility.

HTML responses carry `Cache-Control: no-store` headers so browsers never serve a
stale management page.

Mutating routes validate `model_id` / `repo_id` / `filename` with strict regexes and
`Path.resolve()` / `is_relative_to` checks to prevent path traversal.

**Before exposing beyond the LAN**, add at minimum a reverse proxy with HTTPS and
consider HTTP basic auth or OIDC for the management routes.

Known limitations:
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
