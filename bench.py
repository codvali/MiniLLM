#!/usr/bin/env python3
"""Benchmark Ollama vs MiniLLM — tok/s + latență, același prompt.

Usage:
  python3 bench.py --base http://127.0.0.1:11434 --model qwen3:14b
  python3 bench.py --base http://127.0.0.1:11435 --model qwen3:14b \
      --runs 3 --predict 400

Două rulări (cu/fără draft în minillm.json) → diferența = câștigul
speculative decoding. stdlib only.
"""
import argparse
import json
import statistics
import sys
import time
import urllib.request

if hasattr(sys.stdout, "reconfigure"):  # console fără UTF-8 (Windows)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# prompt reprezentativ: sinteză de articol din dosar (dimensiune reală)
PROMPT = """Ești redactor. Scrie un articol de știri în română pe baza dosarului.

DOSAR:
Eveniment: Primarul Timișoarei a participat la un miting dedicat
comemorării Revoluției din 1989, organizat de un partid politic local.
Persoane: Dominic Fritz (primar), membri ai partidului, participanți.
Locuri: Timișoara, Piața Victoriei, Piața Operei.
Cronologie: eveniment anunțat pentru dimineață, discurs programat,
apoi depuneri de coroane la monumentele dedicate eroilor Revoluției.
Declarații: sursele menționează mesaje despre libertate și memoria
celor care au murit în decembrie 1989.
Context: 16-22 decembrie 1989 — revoltele de la Timișoara au declanșat
căderea regimului comunist în România.

SURSE (extrase):
- Outlet A: „manifestație dedicată eroilor Revoluției din Timișoara"
- Outlet B: „primarul a rostit un discurs în Piața Victoriei"
- Outlet C: „depuneri de coroane și momente de reculegere"

Scrie articolul: lead cu cine/ce/unde/când, corp cu detalii atribuite,
context istoric scurt. Fără speculații. Doar fapte din dosar."""


def one_call(base, model, prompt, n_predict, timeout=2400):
    body = {"model": model, "prompt": prompt, "stream": False,
            "options": {"num_predict": n_predict, "temperature": 0.2}}
    req = urllib.request.Request(
        base.rstrip("/") + "/api/generate",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read())
    wall = time.time() - t0
    evals = d.get("eval_count") or 0
    eval_ns = d.get("eval_duration") or 0
    toks = evals / (eval_ns / 1e9) if eval_ns else (evals / wall if wall else 0)
    load_ns = d.get("load_duration") or 0
    prompt_ns = d.get("prompt_eval_duration") or 0
    return {"wall_s": wall, "tok_s": toks, "eval_count": evals,
            "load_s": load_ns / 1e9, "prompt_s": prompt_ns / 1e9,
            "response_len": len(d.get("response", ""))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="ex. http://127.0.0.1:11435")
    ap.add_argument("--model", required=True)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--predict", type=int, default=400)
    ap.add_argument("--prompt-file", help="alt prompt din fișier")
    args = ap.parse_args()
    prompt = (open(args.prompt_file, encoding="utf-8").read()
              if args.prompt_file else PROMPT)

    print(f"target={args.base} model={args.model} "
          f"runs={args.runs} predict={args.predict}")
    print(f"prompt={len(prompt)} chars")
    print("warmup...")
    try:
        w = one_call(args.base, args.model, prompt, 32)
        print(f"  warmup ok (load={w['load_s']:.1f}s)")
    except Exception as e:
        print(f"WARMUP FAILED: {e}")
        sys.exit(1)

    walls, toks = [], []
    for i in range(args.runs):
        r = one_call(args.base, args.model, prompt, args.predict)
        walls.append(r["wall_s"])
        toks.append(r["tok_s"])
        print(f"run{i+1}: wall={r['wall_s']:.1f}s "
              f"eval={r['eval_count']}tok "
              f"decode={r['tok_s']:.1f}tok/s "
              f"prompt_eval={r['prompt_s']:.1f}s")
    print(f"→ median decode: {statistics.median(toks):.1f} tok/s | "
          f"median wall: {statistics.median(walls):.1f}s")


if __name__ == "__main__":
    main()
