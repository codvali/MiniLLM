# MiniLLM

**One Python file. Zero dependencies. Both APIs.**

A thin orchestrator over `llama-server` (llama.cpp) that speaks
**both** the Ollama API and the OpenAI API — and gives you the full
llama.cpp surface (speculative decoding, per-model tuning, hard
resource budgets) in a single file.

[![License](https://img.shields.io/badge/license-personal%20use-blue)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)]()
[![llama.cpp](https://img.shields.io/badge/backend-llama.cpp-green)]()
[![deps](https://img.shields.io/badge/deps-0-brightgreen)]()

### Try it

```bash
cp minillm.example.json minillm.json   # set llama_server + models
./minillm.py --config minillm.json     # that's it — server on :11435
```

```bash
# Ollama-style
curl http://127.0.0.1:11435/api/generate -d '{
  "model": "qwen3:14b", "prompt": "hello", "stream": false }'

# OpenAI-style (streaming works)
curl http://127.0.0.1:11435/v1/chat/completions -d '{
  "model": "qwen3:14b",
  "messages": [{"role": "user", "content": "Why is the sky blue?"}]
}'

# Drop a GGUF into auto_dir → served as a model in ~15s
./pull.sh Qwen/Qwen3-8B-GGUF Qwen3-8B-Q4_K_M.gguf
```

## What it is

Most local-LLM servers trade control for convenience: the moment
you need speculative decoding, per-model context sizes, or a hard
RAM/CPU budget on a shared box, the knobs you want aren't exposed.

MiniLLM is the opposite trade: one Python file, zero dependencies,
that gives you the full llama.cpp surface with a two-API frontend:

- **Speculative decoding** per model — pair any model with a small
  draft and get **+55% tok/s** for free (measured, see below)
- **Per-model `ctx`, `batch`, `threads`, `ngl`** — fixed at spawn,
  not negotiated per request
- **Explicit resource control** — `max_loaded` + `ram_budget_gb`
  with LRU eviction that never kills a request mid-flight;
  one systemd service = one cgroup = one hard CPU quota for all
  inference
- **`auto_dir`** — drop a `.gguf` in a folder, it's a served model
  in ~15 s. No pull, no config, no restart
- **Two APIs at once** — Ollama (`/api/generate`, `/api/ps`,
  `/api/tags`) and OpenAI (`/v1/chat/completions`, streaming):
  existing clients, scripts, and tools just work
- **Your build, your flags** — point `llama_server` at any
  `llama.cpp` build; use `extra_args` for anything else
- **Already have models pulled?** The `"ollama"` config key resolves
  blobs straight from an existing Ollama library — zero re-download
- **Vision & tools** — `images[]` support, `--mmproj`, chat template
  kwargs, whatever the model needs

Compared to running raw `llama-server` per model, MiniLLM adds the
management layer: on-demand spawn, keep-alive TTL, idle unload,
health checks, and a single front door.

### Speed, measured

From our production box (dual Xeon, CPU-only, `Q4_K_M`):

- **Speculative decoding** (14B model + 0.6B draft): **+55% tok/s**
- **Native build matters more than anything**: `llama-server` built
  with `-DGGML_NATIVE=ON` (AVX2/FMA/F16C) vs a generic scalar build
  was worth more than 2× on prompt processing — same hardware,
  same cores. MiniLLM lets you use *your* build, so this is yours
  to keep.
- `bench.py` measures prompt-eval and decode tok/s on your own
  hardware — numbers over vibes.

## Requirements

- Linux, Python ≥ 3.9
- a `llama-server` binary (`cmake -B build && cmake --build build`)
- GGUF models — your own files, or reused Ollama blobs

## Quick start

```bash
cp minillm.example.json minillm.json   # edit paths + models
mkdir -p models
./minillm.py --config minillm.json     # or: --dry-run to validate
```

Systemd (user service, survives logout with `loginctl enable-linger`):

```bash
mkdir -p ~/.config/systemd/user
cp minillm.service ~/.config/systemd/user/
systemctl --user enable --now minillm
```

Cap its CPU so inference can never starve the rest of the box:

```bash
systemctl --user edit minillm   # add: [Service] CPUQuota=800%
```

## Auto-discovery

Drop a `.gguf` into `auto_dir` (`models/` by default):

```bash
./pull.sh Qwen/Qwen3-8B-GGUF Qwen3-8B-Q4_K_M.gguf
```

Within ~15 s it appears as `qwen3-8b-q4_k_m` on an auto-assigned port
(`auto_port_base`+). `mmproj-*.gguf` files are skipped; files already
in `models` config are deduplicated by path. Sensible defaults apply
(`defaults{}`); set `ctx`/`port` explicitly in config to override.

## API

| Endpoint | Shape | Notes |
|---|---|---|
| `GET /health` | — | `{"ok": true}` |
| `GET /api/tags` | Ollama | all models incl. auto-discovered |
| `GET /api/ps` | Ollama | loaded models + real RSS from `/proc` |
| `POST /api/generate` | Ollama | `model`, `prompt`, `system`, `stream`, `images[]` (vision models), `keep_alive` |
| `GET /v1/models` | OpenAI | all models |
| `POST /v1/chat/completions` | OpenAI | verbatim passthrough, SSE streaming |

Any OpenAI-compatible client works out of the box:

```python
from openai import OpenAI
c = OpenAI(base_url="http://127.0.0.1:11435/v1", api_key="x")
c.chat.completions.create(model="qwen3:14b",
    messages=[{"role": "user", "content": "hi"}])
```

## Model config keys

```jsonc
"name": {
  "gguf": "/abs/path.gguf",      // literal file …
  "ollama": "qwen3:14b",         // … or reuse an Ollama blob
  "port": 12502,                 // fixed upstream port
  "ctx": 16384, "threads": 8,    // per-model overrides
  "est_gb": 12.0,                // for the RAM budget
  "vision": true,                // allow images[]
  "draft": {"ollama": "qwen3:0.6b"},  // speculative decoding
  "draft_max": 16, "draft_min": 4,
  "extra_args": ["--mmproj", "..."],  // anything else llama-server takes
  "chat_template_kwargs": {"enable_thinking": false}
}
```

## Resource safety

- `max_loaded` + `ram_budget_gb`: LRU eviction, never mid-flight
  (`inflight` requests are protected)
- `keep_alive_s`: idle models unload themselves; per-request
  `keep_alive` overrides (`0` = unload now, `-1` = pinned)
- the daemon itself is one process — `systemd CPUQuota` gives you a
  real, hard ceiling for *all* inference at once

## Files

| File | Purpose |
|---|---|
| `minillm.py` | the whole daemon |
| `minillm.example.json` | annotated config template |
| `minillm.service` | systemd user unit |
| `pull.sh` | download a GGUF from HuggingFace into `auto_dir` |
| `bench.py` | per-model prompt/gen benchmark |
| `ts_forward.py` | tiny TCP forwarder (expose on Tailscale/LAN) |

## License

Free for **personal, non-commercial use** — see [LICENSE](LICENSE).
Commercial use requires a separate license from the author.
