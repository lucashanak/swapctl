# swapctl Backend Registry

Backends are llama.cpp builds with different acceleration paths.
swapctl picks one per model based on `model_predicate` + user choice in Matrix UI.

Hardware target: **AMD Strix Halo gfx1151 (Ryzen AI MAX+ 395, 96 GB UMA)**.

## Registry location

- `/var/lib/swapctl/registry.json` → `backends` dict
- swapctl serves it via `GET /manage/api/backends`
- Matrix UI uses it to populate per-model dropdown

## Backend definition schema

```json
"backend-id": {
  "binary_path": "/usr/local/bin/llama-server-…",
  "builtin": false,
  "caps": ["vulkan|hip|rock", "mtp", "gemma4-assistant"],
  "default_flags": ["--spec-type draft-mtp", …],
  "incompatible_flag_patterns": ["--parallel\\s+([2-9]|\\d{2,})"],
  "label": "Human Readable",
  "model_predicate": "has_mtp_heads | is_gemma4_main | null",
  "required_flags": ["--parallel 1"],
  "version_hint": "source/commit/sdk info"
}
```

- `caps` → unstructured tags, used by `auto_bench_backends` to detect spec-capable
- `model_predicate` → name from `app/backends_registry.py:PREDICATES`. `null` = applies to all
- `default_flags` → appended to model cmd when this backend is applied
- `incompatible_flag_patterns` → regex; matching flags get stripped before injection
- `required_flags` → enforced; existing values replaced

## Predicates

- `has_mtp_heads(model)` → True if model has MTP heads OR `draft_model` references an `-mtp-` file
- `is_moe(model)` → True if `'moe'` tag present
- `is_gemma4_main(model)` → True for `gemma-4-*-it` (excluding the assistant draft companions)

## Currently registered backends (2026-05-21)

| ID | Binary | Source | Predicate | Notes |
|---|---|---|---|---|
| `vulkan` | `/usr/local/bin/llama-server-vulkan` → `/opt/llama.cpp/build-vulkan-old/bin/llama-server` | llama.cpp mainline b9209 Vulkan | none | builtin default |
| `vulkan-mtp` | `/usr/local/bin/llama-server-vulkan-mtp` | llama.cpp mainline b9206 (2dff7ff8) Vulkan | `has_mtp_heads` | original am17an PR #22673 merge HEAD; b28a2f3 alternative as `.b28a2f3.bak` |
| `vulkan-gemma4-assistant` | `/usr/local/bin/llama-server-vulkan-gemma4` | llama.cpp am17an PR #23398 (`a03120c`) Vulkan | `is_gemma4_main` | Gemma 4 MTP assistant path; needs am17an's MTP draft GGUF |
| `hip` | `/usr/local/bin/llama-server-hip` | (legacy install, pre-MTP) | none | older builtin; no MTP |
| `hip-mtp` | `/usr/local/bin/llama-server-hip-mtp` | llama.cpp mainline b28a2f3 HIP, ROCm 7.2.3 | `has_mtp_heads` | spec-mtp on stock ROCm; faster for low-accept workloads |
| `rock` | `/usr/local/bin/llama-server-rock` → `/opt/llama.cpp-master/build-rock/bin/llama-server` | llama.cpp mainline b28a2f3 HIP, TheRock SDK (`/opt/rocm-rock`) | none | best non-MTP baseline; tuned hipBLASLt |
| `rock-mtp` | same wrapper as `rock` | same | `has_mtp_heads` | MTP variant of rock |

## Performance snapshot (2026-05-21, Qwen 3.6 35B-A3B Q4_0 + MTP draft, ctx 4096, fa on, np=1)

| Backend | no-MTP gen t/s | +MTP gen t/s | accept | Notes |
|---|---|---|---|---|
| vulkan | 40.4 | — | — | baseline plain |
| vulkan-mtp | — | 37.9 (CLI), 93.6 (swap UI q8 KV ctx 65536) | 52-85% | wins at high accept rate |
| hip-mtp | 26.6 | 68.9 | 57% | wins at low accept rate |
| rock | 61.1 | — | — | **2.3× hip baseline** via TheRock hipBLASLt |
| rock-mtp | — | 68.7 | 55% | MTP adds only 12% on top of fast baseline |

Multi-user (np=8, par8 aggregate, 200 tok/req): vulkan ~163, hip ~169, vulkan-mtp ~91, hip-mtp ~85.

## When each backend wins

- **vulkan-mtp** → integrated MTP heads (`-mtp-iq4xs-q8nextn` etc.) with high accept (>80%); single-user chat with predictable prompts
- **hip-mtp** → MTP with separate draft on Q4_0/Q4_K_M; low-accept workloads
- **rock** → any non-MTP model; or MTP fallback when accept is too low to justify spec overhead
- **rock-mtp** → MTP on TheRock SDK; saturated at ~68 t/s but lower variance
- **vulkan-gemma4-assistant** → only `gemma-4-*-it` main models paired with am17an MTP draft

## Build sources

| Build dir on llm-gpu | Source | Backend(s) |
|---|---|---|
| `/opt/llama.cpp/build-vulkan-old` | b9209 Vulkan | `vulkan` |
| `/opt/llama.cpp-master/build` | mainline b28a2f3 Vulkan | (alternative for vulkan-mtp) |
| `/opt/llama.cpp-master/build-hip` | mainline b28a2f3 HIP, stock ROCm 7.2.3 | `hip-mtp` |
| `/opt/llama.cpp-master/build-rock` | mainline b28a2f3 HIP, TheRock SDK | `rock`, `rock-mtp` |
| `/opt/llama.cpp-am17an-g4` | am17an PR #23398 (gemma4-mtp) | `vulkan-gemma4-assistant` |

## Adding a new backend

1. Build llama-server with desired flags
2. Install binary (or wrapper script with LD_LIBRARY_PATH) at `/usr/local/bin/llama-server-<id>`
3. Edit `/var/lib/swapctl/registry.json` → add entry to `backends`
4. `systemctl restart swapctl llama-swap`
5. Verify via `curl ai.hanaktech.org/manage/api/backends`

## Updating existing backend (binary refresh)

1. Build new llama-server (typically `cmake --build … --target llama-server`)
2. `cp build/bin/llama-server /usr/local/bin/llama-server-<id>` (preserve any `.bak`)
3. `cp build/bin/lib*.so /usr/local/lib/ && ldconfig` (only if not using RPATH)
4. `systemctl restart llama-swap` to pick up; **no swapctl restart needed**

## Build recipe templates

### Vulkan
```bash
cd /opt/llama.cpp-master
cmake -B build \
  -DGGML_VULKAN=ON \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_BUILD_WEBUI=OFF \
  -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON
cmake --build build -j 1 --target llama-server
```

### HIP (stock ROCm)
```bash
cmake -B build-hip \
  -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1151 \
  -DCMAKE_C_COMPILER=/opt/rocm/llvm/bin/amdclang \
  -DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/amdclang++ \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_BUILD_WEBUI=OFF \
  -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON
```

### Rock (TheRock SDK)
```bash
ROCM_PATH=/opt/rocm-rock CMAKE_PREFIX_PATH=/opt/rocm-rock cmake -B build-rock \
  -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1151 \
  -DCMAKE_HIP_COMPILER_ROCM_ROOT=/opt/rocm-rock \
  -DCMAKE_C_COMPILER=/opt/rocm-rock/lib/llvm/bin/amdclang \
  -DCMAKE_CXX_COMPILER=/opt/rocm-rock/lib/llvm/bin/amdclang++ \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_BUILD_WEBUI=OFF \
  -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON

# Runtime needs LD_LIBRARY_PATH=/opt/rocm-rock/lib (wrapper script).
```

### Notes & gotchas

- Always use `-j 1` for shader build steps to avoid OOM kills on gfx1151 builds
- `--no-mmap` mandatory on UMA otherwise CPU-side load slows everything
- Quantized KV cache (`--cache-type-k q8_0 --cache-type-v q8_0`) breaks MTP graph on am17an PR #23398 — leave default f16 for Gemma 4 MTP
- `--threads 8` (lower than default 16) gives ~+3% on MTP — default 16 contends with dispatch thread
- `-b 2048 -ub 512` recommended by hogeheer guide but our measurements show ~no effect

## Bench logic in swapctl

`app/benchmark.py:auto_bench_backends(model_id)`:
1. Start with `["vulkan"]` (unless model has baked-in MTP heads where vulkan crashes loading it)
2. Add every spec-capable backend whose predicate matches (caps contains `mtp`/`draft`/`gemma4-assistant`)
3. Return list ordered by registry insertion order

Auto mode = bench all returned; Manual mode = bench user-selected from compatible set.

## Tracking upstream

- **am17an PR #23398** (Gemma 4 MTP) — pending mainline merge; once landed, drop separate vulkan-gemma4-assistant backend and use `vulkan-mtp` + `gemma4-assistant` cap
- **hogeheer499-commits/strix-halo-guide** — community benchmarks; watch for new community-tested configs
- **vLLM gfx1151 + Gemma 4 MTP** — blocked on HIP graph capture (vLLM #32180); not viable yet
- **lhl/llama.cpp@rocm-wmma-tune** — rocWMMA-tuned ROCm fork; incompatible with mainline MTP (older base) but useful for non-MTP workloads if rebased
