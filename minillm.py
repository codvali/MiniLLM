#!/usr/bin/env python3
"""MiniLLM — thin Ollama+OpenAI-compatible orchestrator over llama-server.

Why it exists: most LLM frontends abstract llama.cpp away and hide
the flags that matter on a shared box. MiniLLM exposes them:
  - speculative decoding  (llama-server -md/--model-draft)  ~1.5-3x faster
  - num_ctx / num_batch per MODEL, fixed at spawn (not per-request)
  - explicit LRU residency + RAM budget
  - any llama.cpp build you compile yourself, the day it lands

Drop-in for the subset of the Ollama API our stack uses:
  GET  /api/tags               -> {"models":[{"name": ...}]}
  POST /api/generate           -> Ollama-shaped response (stream + images[])
  GET  /api/ps                 -> loaded models + real RSS from /proc
  GET  /health                 -> {"ok": true}
  GET  /v1/models              -> OpenAI model list
  POST /v1/chat/completions    -> OpenAI passthrough (stream supported)

Zero dependencies: Python 3.9+ stdlib only.

Auto-discovery:
  drop any .gguf into "auto_dir" and it becomes a servable model
  within ~15s — name derived from the filename, port auto-assigned
  from "auto_port_base". mmproj-*.gguf files are skipped; files
  already referenced in "models" are deduplicated by path.

GGUF resolution:
  each model config points either at a literal "gguf" path or at an
  "ollama" model name — resolved through the Ollama manifest -> blob,
  so models already pulled for Ollama are reused, no re-download.
"""
import argparse
import http.client
import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

OLLAMA_MODEL_MEDIATYPE = "application/vnd.ollama.image.model"


# ---------------------------------------------------------------- config

def load_config(path):
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)
    cfg.setdefault("listen", "127.0.0.1")
    cfg.setdefault("port", 11435)
    cfg.setdefault("llama_server", "llama-server")
    cfg.setdefault("ollama_dir", os.path.expanduser("~/.ollama/models"))
    cfg.setdefault("max_loaded", 3)
    cfg.setdefault("ram_budget_gb", 0)          # 0 = off
    cfg.setdefault("keep_alive_s", 1800)        # default idle TTL
    cfg.setdefault("load_timeout_s", 300)
    cfg.setdefault("log_dir", os.path.expanduser("~/.cache/minillm"))
    cfg.setdefault("defaults", {})
    cfg.setdefault("models", {})
    cfg.setdefault("auto_dir", "")
    cfg.setdefault("auto_port_base", 12600)
    if not cfg["models"]:
        raise SystemExit("config has no models")
    return cfg


def resolve_ollama_blob(ollama_dir, model_name):
    """'qwen3:14b' -> ~/.ollama/models/blobs/sha256-<hex> via manifest."""
    name, _, tag = model_name.partition(":")
    tag = tag or "latest"
    manifest = os.path.join(
        ollama_dir, "manifests", "registry.ollama.ai",
        "library", name, tag)
    if not os.path.exists(manifest):
        # non-library namespace (user-pushed / custom): search
        ns_root = os.path.join(
            ollama_dir, "manifests", "registry.ollama.ai")
        manifest = None
        for root, _dirs, files in os.walk(ns_root):
            if tag in files and os.path.basename(root) == name:
                manifest = os.path.join(root, tag)
                break
        if manifest is None:
            raise FileNotFoundError(
                f"ollama manifest not found for {model_name}")
    with open(manifest, encoding="utf-8") as f:
        doc = json.load(f)
    for layer in doc.get("layers", []):
        if layer.get("mediaType") == OLLAMA_MODEL_MEDIATYPE:
            digest = layer["digest"].split(":", 1)[-1]
            blob = os.path.join(ollama_dir, "blobs", f"sha256-{digest}")
            if not os.path.exists(blob):
                raise FileNotFoundError(
                    f"blob missing for {model_name}: {blob}")
            return blob
    raise FileNotFoundError(
        f"no model layer in manifest for {model_name}")


def parse_keep_alive(v):
    """Ollama keep_alive: int seconds, '30m', '1h', 0=unload now, <0=never."""
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip().lower()
    mult = {"ms": 0.001, "s": 1, "m": 60, "h": 3600}
    for suf, m in sorted(mult.items(), key=lambda x: -len(x[0])):
        if s.endswith(suf):
            return float(s[:-len(suf)] or 0) * m
    try:
        return float(s)
    except ValueError:
        return None


def model_gguf_path(cfg, mcfg):
    if mcfg.get("gguf"):
        return mcfg["gguf"]
    if mcfg.get("ollama"):
        return resolve_ollama_blob(cfg["ollama_dir"], mcfg["ollama"])
    raise ValueError("model needs 'gguf' or 'ollama'")


# ------------------------------------------------------------- runner

def proc_rss_bytes(pid):
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError):
        pass
    return 0


class Runner:
    """One llama-server subprocess for one model."""

    def __init__(self, name, mcfg, gguf, cfg):
        self.name = name
        self.port = mcfg["port"]
        self.proc = None
        self.last_used = 0.0
        self.loading = False
        self.ttl_override = None  # seconds; per-request keep_alive
        self.inflight = 0         # active requests — reaper/evict must skip
        self.lock = threading.RLock()
        self._mcfg = mcfg
        self._gguf = gguf
        self._cfg = cfg

    # ---- lifecycle -------------------------------------------------
    def cmdline(self):
        c, d = self._mcfg, self._cfg["defaults"]
        get = lambda k: c.get(k, d.get(k))
        cmd = [self._cfg["llama_server"],
               "-m", self._gguf,
               "--host", "127.0.0.1",
               "--port", str(self.port),
               "-c", str(get("ctx") or 8192),
               "-b", str(get("batch") or 512),
               "-ub", str(get("ubatch") or 256),
               "-t", str(get("threads") or 0) or "0",
               "-ngl", str(get("ngl") or 0),
               "--jinja", "--no-webui"]
        fa = get("flash_attn")
        if fa:
            cmd += ["--flash-attn", "on"]
        for k in ("cache_type_k", "cache_type_v"):
            v = get(k)
            if v:
                cmd += [f"--cache-type-{k.split('_')[-1]}", str(v)]
        draft = c.get("draft")
        if draft:  # speculative decoding — what Ollama can't do
            cmd += ["-md", model_gguf_path(self._cfg, draft),
                    "--spec-draft-n-max", str(c.get("draft_max", 16)),
                    "--spec-draft-n-min", str(c.get("draft_min", 4))]
            if c.get("draft_ngl"):
                cmd += ["-ngld", str(c["draft_ngl"])]
        extra = c.get("extra_args") or d.get("extra_args") or []
        cmd += [str(x) for x in extra]
        return cmd

    def healthy(self):
        if not self.proc or self.proc.poll() is not None:
            return False
        try:
            conn = http.client.HTTPConnection(
                "127.0.0.1", self.port, timeout=3)
            conn.request("GET", "/health")
            return conn.getresponse().status == 200
        except OSError:
            return False

    def ensure_up(self):
        with self.lock:
            if self.healthy():
                self.last_used = time.time()
                return
            if self.proc:  # dead — clean corpse
                self.stop(force=True)
            self.loading = True
            os.makedirs(self._cfg["log_dir"], exist_ok=True)
            logf = open(os.path.join(
                self._cfg["log_dir"], f"{self.name}.log"), "ab")
            cmd = self.cmdline()
            print(f"[minillm] spawn {self.name}: {' '.join(cmd)}",
                  flush=True)
            self.proc = subprocess.Popen(
                cmd, stdout=logf, stderr=subprocess.STDOUT)
            self.last_used = time.time()
            deadline = self.last_used + self._cfg["load_timeout_s"]
            while time.time() < deadline:
                if self.proc.poll() is not None:
                    raise RuntimeError(
                        f"{self.name}: llama-server exited "
                        f"{self.proc.returncode} — see log")
                if self.healthy():
                    self.last_used = time.time()
                    self.loading = False
                    return
                time.sleep(0.5)
            self.stop(force=True)
            raise RuntimeError(f"{self.name}: health timeout")

    def stop(self, force=False):
        with self.lock:
            if not self.proc:
                return
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
            self.proc = None
            self.loading = False

    def rss(self):
        return proc_rss_bytes(self.proc.pid) if self.proc else 0


# ------------------------------------------------------------- manager

class Manager:
    def __init__(self, cfg):
        self.cfg = cfg
        self.runners = {}
        self.mu = threading.Lock()
        for name, mcfg in cfg["models"].items():
            gguf = model_gguf_path(cfg, mcfg)
            self.runners[name] = Runner(name, mcfg, gguf, cfg)
        self.discover()
        threading.Thread(target=self._reaper, daemon=True).start()

    def discover(self):
        """Auto-inregistreaza *.gguf aruncate in auto_dir.

        Numele modelului = fisierul normalizat
        ('Qwen3-8B-Q4_K_M.gguf' -> 'qwen3-8b-q4_k_m').
        mmproj-*.gguf si fisierele deja referite in config
        sunt ignorate. Portul e alocat din auto_port_base.
        """
        adir = self.cfg.get("auto_dir")
        if not adir or not os.path.isdir(adir):
            return
        with self.mu:
            known = {r._gguf for r in self.runners.values()}
            used_ports = {r.port for r in self.runners.values()}
            port = int(self.cfg.get("auto_port_base", 12600))
            for f in sorted(os.listdir(adir)):
                if not f.lower().endswith(".gguf"):
                    continue
                if f.lower().startswith("mmproj"):
                    continue
                p = os.path.join(adir, f)
                if not os.path.isfile(p) or p in known:
                    continue
                stem = os.path.splitext(f)[0]
                name = re.sub(r"[^a-z0-9.]+", "-",
                              stem.lower()).strip("-")
                if not name or name in self.runners:
                    continue
                while port in used_ports:
                    port += 1
                used_ports.add(port)
                try:
                    est = os.path.getsize(p) / 1024 ** 3
                except OSError:
                    est = 0
                mcfg = {"gguf": p, "port": port, "est_gb": est,
                        "auto": True}
                self.runners[name] = Runner(name, mcfg, p, self.cfg)
                print(f"[minillm] auto-discovered {name} "
                      f"({f}, {est:.1f}G, port {port})", flush=True)

    def loaded(self):
        return [r for r in list(self.runners.values())
                if r.proc and r.proc.poll() is None]

    def _evict_for(self, want):
        """LRU evict until max_loaded / ram_budget have room.
        Runners serving a request are never evicted mid-flight."""
        loaded = [r for r in self.loaded()
                  if r.name != want and r.inflight == 0]
        # by resident count
        while len(loaded) >= self.cfg["max_loaded"]:
            victim = min(loaded, key=lambda r: r.last_used)
            print(f"[minillm] evict {victim.name} (LRU)", flush=True)
            victim.stop()
            loaded.remove(victim)
        # by RAM budget
        budget = self.cfg["ram_budget_gb"] * 1024 ** 3
        if budget:
            mcfg = self.cfg["models"][want]
            est = mcfg.get("est_gb", 0) * 1024 ** 3
            while (sum(r.rss() for r in loaded) + est > budget
                   and loaded):
                victim = min(loaded, key=lambda r: r.last_used)
                print(f"[minillm] evict {victim.name} (RAM)", flush=True)
                victim.stop()
                loaded.remove(victim)

    def runner_for(self, name):
        r = self.runners.get(name) or next(
            (r for n, r in list(self.runners.items())
             if n.startswith(name)), None)
        if r is None:
            self.discover()
            r = self.runners.get(name) or next(
                (r for n, r in list(self.runners.items())
                 if n.startswith(name)), None)
        if r is None:
            raise KeyError(name)
        with self.mu:
            self._evict_for(r.name)
        r.ensure_up()
        return r

    def _reaper(self):
        while True:
            time.sleep(15)
            self.discover()
            now = time.time()
            for r in list(self.runners.values()):
                if not (r.proc and r.proc.poll() is None):
                    continue
                if r.loading:
                    continue
                if r.inflight > 0:
                    continue  # never unload mid-request
                ka = (r.ttl_override if r.ttl_override is not None
                      else r._mcfg.get(
                          "keep_alive_s", self.cfg["keep_alive_s"]))
                if ka >= 0 and now - r.last_used > ka:
                    print(f"[minillm] idle unload {r.name}", flush=True)
                    r.stop()


# ------------------------------------------------------ ollama bridge

# ollama generate options -> llama-server /completion fields
OPT_MAP = {
    "num_predict": "n_predict",
    "temperature": "temperature",
    "top_p": "top_p",
    "min_p": "min_p",
    "top_k": "top_k",
    "repeat_penalty": "repeat_penalty",
    "seed": "seed",
    "stop": "stop",
    "presence_penalty": "presence_penalty",
    "frequency_penalty": "frequency_penalty",
}


def to_llama_completion(body):
    opts = body.get("options") or {}
    comp = {"prompt": body["prompt"], "stream": False,
            "cache_prompt": True}
    for okey, lkey in OPT_MAP.items():
        if okey in opts:
            comp[lkey] = opts[okey]
    if "n_predict" not in comp:
        comp["n_predict"] = 2048
    if body.get("format") == "json":
        comp["response_format"] = {"type": "json_object"}
    return comp


# chat-mode: Ollama renders each model's Modelfile TEMPLATE around
# /api/generate prompts — bare /completion skips that and instruct
# models (minicpm, qwen) degenerate into babble. /v1/chat/completions
# makes llama-server apply the GGUF-embedded chat template itself.
CHAT_OPT_MAP = {
    "num_predict": "max_tokens",
    "temperature": "temperature",
    "top_p": "top_p",
    "min_p": "min_p",
    "top_k": "top_k",
    "repeat_penalty": "repeat_penalty",
    "seed": "seed",
    "stop": "stop",
    "presence_penalty": "presence_penalty",
    "frequency_penalty": "frequency_penalty",
}


def to_llama_chat(body, system=None):
    opts = body.get("options") or {}
    msgs = ([{"role": "system", "content": system}] if system else [])
    imgs = body.get("images") or []
    if imgs:
        content = []
        for b in imgs:
            b = str(b)
            url = b if b.startswith("data:") else \
                    "data:image/jpeg;base64," + b
            content.append({"type": "image_url",
                            "image_url": {"url": url}})
        content.append({"type": "text",
                        "text": body["prompt"]})
        msgs.append({"role": "user", "content": content})
    else:
        msgs.append({"role": "user",
                     "content": body["prompt"]})
    req = {"messages": msgs, "stream": False,
           "cache_prompt": True, "max_tokens": 2048}
    for okey, lkey in CHAT_OPT_MAP.items():
        if okey in opts:
            req[lkey] = opts[okey]
    if body.get("format") == "json":
        req["response_format"] = {"type": "json_object"}
    if body.get("think") is False:
        # qwen3-style: /no_think via template kwarg
        req["chat_template_kwargs"] = {"enable_thinking": False}
    return req


def from_llama_chat(d):
    try:
        msg = d["choices"][0]["message"]
        # reasoning models (minicpm5, qwen3-think) may put everything
        # in reasoning_content when thinking is enabled — surface it
        return msg.get("content") or msg.get("reasoning_content") or ""
    except (KeyError, IndexError, TypeError):
        return ""


def stream_piece_to_ollama(name, d, chat):
    """One upstream SSE data-object -> one Ollama NDJSON line dict."""
    if chat:
        try:
            delta = d["choices"][0]["delta"]
            piece = (delta.get("content")
                     or delta.get("reasoning_content") or "")
        except (KeyError, IndexError, TypeError):
            piece = ""
        stop = bool((d.get("choices") or [{}])[0].get("finish_reason"))
    else:
        piece = d.get("content", "")
        stop = bool(d.get("stop"))
    out = {"model": name,
           "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                       time.gmtime()),
           "response": piece, "done": False}
    if stop:
        out["done"] = True
        usage = d.get("usage") or {}
        out["eval_count"] = usage.get(
            "completion_tokens", d.get("tokens_predicted", 0))
        out["prompt_eval_count"] = usage.get(
            "prompt_tokens", d.get("tokens_evaluated", 0))
    return out


def to_ollama_response(model, comp):
    timings = comp.get("timings") or {}
    return {
        "model": model,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                    time.gmtime()),
        "response": comp.get("content", ""),
        "done": True,
        "done_reason": "stop" if comp.get("stop") else "length",
        "eval_count": comp.get("tokens_predicted", 0),
        "prompt_eval_count": comp.get("tokens_evaluated", 0),
        "eval_duration": int(
            (comp.get("tokens_predicted", 0)
             / max(timings.get("predicted_per_second", 1), 1))
            * 1e9),
        "prompt_eval_duration": int(
            timings.get("prompt_ms", 0) * 1e6),
        "total_duration": int(
            timings.get("prompt_ms", 0) * 1e6
            + timings.get("predicted_ms", 0) * 1e6),
    }


# ------------------------------------------------------------- http

class Handler(BaseHTTPRequestHandler):
    mgr = None
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        pass  # quiet — llama-server logs are the real signal

    def _send(self, code, obj):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}")

    def do_GET(self):
        path = urllib.parse.urlparse(self.path).path
        self.mgr.discover()
        if path == "/health":
            return self._send(200, {"ok": True})
        if path == "/api/tags":
            return self._send(200, {"models": [
                {"name": n, "model": n, "size": 0,
                 "modified_at": ""}
                for n in list(self.mgr.runners)]})
        if path == "/api/ps":
            out = []
            for r in self.mgr.loaded():
                out.append({
                    "name": r.name, "model": r.name,
                    "size": r.rss(), "size_vram": 0,
                    "expires_at": time.strftime(
                        "%Y-%m-%dT%H:%M:%SZ", time.gmtime(
                            r.last_used + self.mgr.cfg["keep_alive_s"])),
                })
            return self._send(200, {"models": out})
        if path == "/v1/models":
            return self._send(200, {"object": "list", "data": [
                {"id": n, "object": "model", "created": 0,
                 "owned_by": "minillm"}
                for n in list(self.mgr.runners)]})
        return self._send(404, {"error": "not found"})

    def do_POST(self):
        path = urllib.parse.urlparse(self.path).path
        openai_passthrough = path == "/v1/chat/completions"
        if not openai_passthrough and path != "/api/generate":
            return self._send(404, {"error": "not found"})
        try:
            body = self._body()
        except Exception:
            return self._send(400, {"error": "bad json"})
        if body.get("images"):
            _mc = (self.mgr.cfg.get("models", {}).get(
                body.get("model")) or {})
            if not _mc.get("vision"):
                return self._send(400, {
                    "error": "multimodal unsupported"})
        name = body.get("model")
        try:
            runner = self.mgr.runner_for(name)
        except KeyError:
            return self._send(404, {"error": f"model '{name}' not found"})
        except RuntimeError as e:
            return self._send(503, {"error": str(e)})
        mcfg = self.mgr.cfg["models"].get(runner.name, {})
        chat = mcfg.get("mode", "chat") == "chat"
        if openai_passthrough:
            upstream_path = "/v1/chat/completions"
            upstream_body = dict(body)
            if runner.name != name:
                upstream_body["model"] = runner.name
        elif chat:
            upstream_path = "/v1/chat/completions"
            upstream_body = to_llama_chat(
                body, mcfg.get("system"))
            _ctk = mcfg.get("chat_template_kwargs")
            if _ctk:
                upstream_body.setdefault(
                    "chat_template_kwargs", {}).update(_ctk)
        else:
            upstream_path = "/completion"
            upstream_body = to_llama_completion(body)
        want_stream = bool(body.get("stream"))
        if want_stream:
            upstream_body["stream"] = True
        runner.inflight += 1
        try:
            conn = http.client.HTTPConnection(
                "127.0.0.1", runner.port,
                timeout=self.mgr.cfg.get("gen_timeout_s", 2400))
            conn.request("POST", upstream_path,
                         body=json.dumps(upstream_body),
                         headers={"Content-Type": "application/json"})
            resp = conn.getresponse()
            if openai_passthrough:
                if want_stream and resp.status == 200:
                    return self._relay_sse(resp)
                data = resp.read()
                try:
                    return self._send(resp.status, json.loads(data))
                except ValueError:
                    return self._send(
                        resp.status,
                        {"error": data.decode("utf-8", "replace")})
            if want_stream and resp.status == 200:
                return self._relay_stream(resp, name, chat)
            data = resp.read()
            if resp.status != 200:
                return self._send(resp.status,
                                  {"error": data.decode("utf-8", "replace")})
            if want_stream:
                # upstream refused to stream — degrade to one NDJSON line
                d = json.loads(data)
                if chat:
                    d = {"content": from_llama_chat(d)}
                return self._send(200, to_ollama_response(name, d))
            d = json.loads(data)
            if chat:
                usage = d.get("usage") or {}
                d = {"content": from_llama_chat(d), "stop": True,
                     "tokens_predicted": usage.get("completion_tokens", 0),
                     "tokens_evaluated": usage.get("prompt_tokens", 0)}
            return self._send(200, to_ollama_response(name, d))
        except Exception as e:
            return self._send(503, {"error": f"llama-server: {e}"})
        finally:
            runner.inflight -= 1
            runner.last_used = time.time()
            ka = parse_keep_alive(body.get("keep_alive"))
            if ka is not None:
                runner.ttl_override = ka
                if ka == 0:
                    runner.stop()

    def _relay_sse(self, resp):
        """llama-server SSE -> SSE verbatim (clienti OpenAI)."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        while True:
            chunk = resp.read1(4096) if hasattr(resp, "read1") \
                else resp.read(4096)
            if not chunk:
                break
            self.wfile.write(chunk)
            self.wfile.flush()

    def _relay_stream(self, resp, name, chat):
        """llama-server SSE -> Ollama NDJSON, chunk by chunk."""
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        buf = b""
        try:
            while True:
                chunk = resp.read1(4096) if hasattr(resp, "read1") \
                    else resp.read(4096)
                if not chunk:
                    break
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    line = line.strip()
                    if not line.startswith(b"data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == b"[DONE]":
                        continue
                    try:
                        d = json.loads(payload)
                    except ValueError:
                        continue
                    self.wfile.write(json.dumps(
                        stream_piece_to_ollama(name, d, chat)).encode()
                        + b"\n")
                    self.wfile.flush()
            # closing frame — Ollama clients expect a final done:true
            self.wfile.write(json.dumps({
                "model": name,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                            time.gmtime()),
                "response": "", "done": True}).encode() + b"\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass  # client went away — nothing to relay to


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--check", action="store_true",
                    help="resolve config + ggufs, print plan, exit")
    args = ap.parse_args()
    cfg = load_config(args.config)
    mgr = Manager(cfg)
    if args.check:
        for n, r in list(mgr.runners.items()):
            print(f"{n}: {r._gguf}  -> :{r.port}")
        print("config ok")
        return
    Handler.mgr = mgr
    srv = ThreadingHTTPServer((cfg["listen"], cfg["port"]), Handler)
    print(f"[minillm] listening {cfg['listen']}:{cfg['port']} "
          f"models={list(mgr.runners)}", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        for r in list(mgr.runners.values()):
            r.stop(force=True)


if __name__ == "__main__":
    main()
