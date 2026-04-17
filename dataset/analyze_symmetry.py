"""
UniXcoder Code-Reasoning Similarity Analysis for vcldata.jsonl

Usage:
    uv run python dataset/analyze_symmetry.py --n-samples 10000 --seed 220703

Arguments:
    --n-samples    Number of random samples to analyze (default: all)
    --seed         Random seed (default: 220703)

Input:  dataset/vcldata.jsonl
Output: Statistical report + low-similarity samples
"""

import argparse
import json
import math
import time
import threading
import random

import torch
from transformers import AutoTokenizer, AutoModel
from concurrent.futures import ThreadPoolExecutor, as_completed

print_lock = threading.Lock()


def pprint(msg):
    with print_lock:
        print(msg, flush=True)


_model_cache = {}


def get_model():
    if "model" not in _model_cache:
        model_name = "microsoft/unixcoder-base"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        _model_cache["model"] = (model, tokenizer, device)
    return _model_cache["model"]


def encode(text, max_length=512):
    model, tokenizer, device = get_model()
    tokens = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    ).to(device)
    with torch.no_grad():
        emb = model(**tokens).last_hidden_state[:, 0, :]
    emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.squeeze(0).cpu()


def similarity(code, reason):
    code_emb = encode(code)
    text_emb = encode(reason)
    return (code_emb * text_emb).sum(dim=-1).item()


def process_sample(args):
    i, ex = args
    try:
        sim = similarity(ex.get("func", ""), ex.get("reason", ""))
    except Exception:
        sim = None
    return i, sim


def mean(vals):
    return sum(vals) / len(vals) if vals else 0.0


def std(vals):
    m = mean(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / max(len(vals) - 1, 1))


def pct(vals, p):
    s = sorted(vals)
    idx = int(len(s) * p / 100)
    return s[min(idx, len(s) - 1)]


def main():
    parser = argparse.ArgumentParser(description="UniXcoder similarity analysis")
    parser.add_argument("--n-samples", type=int, default=None,
                        help="Number of random samples to analyze (default: all)")
    parser.add_argument("--seed", type=int, default=220703,
                        help="Random seed (default: 220703)")
    args = parser.parse_args()

    random.seed(args.seed)

    t0 = time.time()

    pprint("Loading dataset/vcldata.jsonl...")
    samples = []
    with open("dataset/vcldata.jsonl") as f:
        for line in f:
            samples.append(json.loads(line))
    pprint(f"  {len(samples):,} samples loaded\n")

    # Subsample if requested
    if args.n_samples is not None and args.n_samples < len(samples):
        samples = random.sample(samples, args.n_samples)
        pprint(f"  Subsampled to {len(samples):,} samples (seed={args.seed})\n")

    pprint("Loading UniXcoder...")
    model, tokenizer, device = get_model()
    pprint(f"  Device: {device}\n")

    total = len(samples)
    results = [None] * total

    pprint(f"Computing {total:,} similarities (8 workers)...")

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(process_sample, (i, ex)): i for i, ex in enumerate(samples)}
        done = 0
        for future in as_completed(futures):
            idx, sim = future.result()
            results[idx] = sim
            done += 1
            if done % 20000 == 0:
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                pprint(f"  {done:,}/{total:,} ({rate:.0f}/sec)")

    elapsed = time.time() - t0
    pprint(f"\nDone in {elapsed:.1f}s ({total / elapsed:.0f}/sec)\n")

    # -------------------------------------------------------------------------
    sims = [s for s in results if s is not None]
    pprint("=" * 60)
    pprint("OVERALL STATISTICS")
    pprint("=" * 60)
    pprint(f"  n     = {len(sims):,}")
    pprint(f"  mean  = {mean(sims):.4f}")
    pprint(f"  std   = {std(sims):.4f}")
    pprint(f"  min   = {min(sims):.4f}")
    pprint(f"  p10   = {pct(sims, 10):.4f}")
    pprint(f"  p25   = {pct(sims, 25):.4f}")
    pprint(f"  p50   = {pct(sims, 50):.4f}")
    pprint(f"  p75   = {pct(sims, 75):.4f}")
    pprint(f"  p90   = {pct(sims, 90):.4f}")
    pprint(f"  p95   = {pct(sims, 95):.4f}")
    pprint(f"  p99   = {pct(sims, 99):.4f}")
    pprint(f"  max   = {max(sims):.4f}")

    # By label
    label_0 = [s for s, ex in zip(sims, samples) if ex.get("label") == "0"]
    label_1 = [s for s, ex in zip(sims, samples) if ex.get("label") == "1"]

    pprint("\n" + "=" * 60)
    pprint("SAFE (label=0)")
    pprint("=" * 60)
    pprint(f"  n     = {len(label_0):,}")
    pprint(f"  mean  = {mean(label_0):.4f}")
    pprint(f"  std   = {std(label_0):.4f}")
    pprint(f"  min   = {min(label_0):.4f}")
    pprint(f"  p50   = {pct(label_0, 50):.4f}")
    pprint(f"  max   = {max(label_0):.4f}")

    pprint("\n" + "=" * 60)
    pprint("VULNERABLE (label=1)")
    pprint("=" * 60)
    pprint(f"  n     = {len(label_1):,}")
    pprint(f"  mean  = {mean(label_1):.4f}")
    pprint(f"  std   = {std(label_1):.4f}")
    pprint(f"  min   = {min(label_1):.4f}")
    pprint(f"  p50   = {pct(label_1, 50):.4f}")
    pprint(f"  max   = {max(label_1):.4f}")

    # Low similarity vulnerable samples
    vuln_with_sim = [(s, samples[i]) for i, s in enumerate(results) if s is not None and samples[i].get("label") == "1"]
    low_vuln = sorted(vuln_with_sim, key=lambda x: x[0])[:50]

    pprint("\n" + "=" * 60)
    pprint("LOWEST SIMILARITY VULNERABLE SAMPLES (label=1, top 30)")
    pprint("=" * 60)
    for sim, ex in low_vuln[:30]:
        reason = ex.get("reason", "")[:100]
        cwe = ex.get("cwe_id")
        func = ex.get("func", "")[:80]
        pprint(f"\n  sim={sim:.4f}  cwe_id={cwe}")
        pprint(f"  reason: {reason}")
        pprint(f"  func:   {func}")

    # Low similarity safe samples
    safe_with_sim = [(s, samples[i]) for i, s in enumerate(results) if s is not None and samples[i].get("label") == "0"]
    low_safe = sorted(safe_with_sim, key=lambda x: x[0])[:20]

    pprint("\n" + "=" * 60)
    pprint("LOWEST SIMILARITY SAFE SAMPLES (label=0, top 20)")
    pprint("=" * 60)
    for sim, ex in low_safe[:20]:
        reason = ex.get("reason", "")[:100]
        cwe = ex.get("cwe_id")
        func = ex.get("func", "")[:80]
        pprint(f"\n  sim={sim:.4f}  cwe_id={cwe}")
        pprint(f"  reason: {reason}")
        pprint(f"  func:   {func}")

    # Distribution bins
    bins = [(0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5),
            (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01)]
    pprint("\n" + "=" * 60)
    pprint("SIMILARITY DISTRIBUTION (all samples)")
    pprint("=" * 60)
    for lo, hi in bins:
        count = sum(1 for s in sims if lo <= s < hi)
        bar = "#" * int(60 * count / len(sims))
        pprint(f"  [{lo:.1f},{hi:.1f})  {count:6,}  {100*count/len(sims):5.1f}%  {bar}")

    pprint("\n" + "=" * 60)
    pprint("SIMILARITY DISTRIBUTION (vulnerable only)")
    pprint("=" * 60)
    for lo, hi in bins:
        count = sum(1 for s in label_1 if lo <= s < hi)
        bar = "#" * int(60 * count / max(len(label_1), 1))
        pprint(f"  [{lo:.1f},{hi:.1f})  {count:6,}  {100*count/max(len(label_1),1):5.1f}%  {bar}")

    pprint("\n" + "=" * 60)
    pprint(f"Analysis complete. Total time: {time.time() - t0:.1f}s")
    pprint("=" * 60)


if __name__ == "__main__":
    main()