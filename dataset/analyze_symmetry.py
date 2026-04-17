"""
UniXcoder Code-Reasoning Similarity Analysis for vcldata.pkl

Usage:
    uv run python dataset/analyze_symmetry.py --n-samples 10000 --seed 220703

Arguments:
    --n-samples    Number of random samples to analyze (default: all)
    --seed         Random seed (default: 220703)

Input:  dataset/vcldata.pkl (list of ExampleFeature)
Output: Statistical report + low-similarity samples
"""

import argparse
import math
import time
import random
import pickle
import sys

import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

sys.path.insert(0, "src")
# from generate_example import ExampleFeature


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


def encode_batch(texts, max_length=512, batch_size=64):
    """Encode a list of texts in batches, returns normalized embeddings."""
    model, tokenizer, device = get_model()
    embeddings = []

    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i + batch_size]
        tokens = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)
        with torch.no_grad():
            emb = model(**tokens).last_hidden_state[:, 0, :]
        emb = emb / emb.norm(dim=-1, keepdim=True)
        embeddings.append(emb.cpu())

    return torch.cat(embeddings, dim=0)


def main():
    parser = argparse.ArgumentParser(description="UniXcoder similarity analysis")
    parser.add_argument("--n-samples", type=int, default=None,
                        help="Number of random samples to analyze (default: all)")
    parser.add_argument("--seed", type=int, default=220703,
                        help="Random seed (default: 220703)")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size for UniXcoder encoding (default: 64)")
    args = parser.parse_args()

    random.seed(args.seed)

    t0 = time.time()

    print("Loading dataset/vcldata.pkl...")
    with open("dataset/vcldata.pkl", "rb") as f:
        samples = pickle.load(f)
    print(f"  {len(samples):,} samples loaded")

    # Subsample if requested
    if args.n_samples is not None and args.n_samples < len(samples):
        samples = random.sample(samples, args.n_samples)
        print(f"  Subsampled to {len(samples):,} samples (seed={args.seed})")

    print("Loading UniXcoder...")
    model, tokenizer, device = get_model()
    print(f"  Device: {device}")

    total = len(samples)

    # Extract all codes and descriptions upfront for batched encoding
    codes = [ex.func for ex in samples]
    reasons = [ex.description for ex in samples]

    print(f"\nEncoding {total:,} codes (batch_size={args.batch_size})...")
    code_embs = encode_batch(codes, batch_size=args.batch_size)

    print(f"Encoding {total:,} descriptions (batch_size={args.batch_size})...")
    desc_embs = encode_batch(reasons, batch_size=args.batch_size)

    # Compute similarities
    print("Computing similarities...")
    sims = ((code_embs * desc_embs).sum(dim=-1)).tolist()

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s ({total / elapsed:.0f}/sec)\n")

    # -------------------------------------------------------------------------
    def mean(vals):
        return sum(vals) / len(vals) if vals else 0.0

    def std(vals):
        m = mean(vals)
        return math.sqrt(sum((v - m) ** 2 for v in vals) / max(len(vals) - 1, 1))

    def pct(vals, p):
        s = sorted(vals)
        idx = int(len(s) * p / 100)
        return s[min(idx, len(s) - 1)]

    print("=" * 60)
    print("OVERALL STATISTICS")
    print("=" * 60)
    print(f"  n     = {len(sims):,}")
    print(f"  mean  = {mean(sims):.4f}")
    print(f"  std   = {std(sims):.4f}")
    print(f"  min   = {min(sims):.4f}")
    print(f"  p10   = {pct(sims, 10):.4f}")
    print(f"  p25   = {pct(sims, 25):.4f}")
    print(f"  p50   = {pct(sims, 50):.4f}")
    print(f"  p75   = {pct(sims, 75):.4f}")
    print(f"  p90   = {pct(sims, 90):.4f}")
    print(f"  p95   = {pct(sims, 95):.4f}")
    print(f"  p99   = {pct(sims, 99):.4f}")
    print(f"  max   = {max(sims):.4f}")

    # By label
    label_0 = [s for s, ex in zip(sims, samples) if ex.label == "0"]
    label_1 = [s for s, ex in zip(sims, samples) if ex.label == "1"]

    print("\n" + "=" * 60)
    print("SAFE (label=0)")
    print("=" * 60)
    print(f"  n     = {len(label_0):,}")
    print(f"  mean  = {mean(label_0):.4f}")
    print(f"  std   = {std(label_0):.4f}")
    print(f"  min   = {min(label_0):.4f}")
    print(f"  p50   = {pct(label_0, 50):.4f}")
    print(f"  max   = {max(label_0):.4f}")

    print("\n" + "=" * 60)
    print("VULNERABLE (label=1)")
    print("=" * 60)
    print(f"  n     = {len(label_1):,}")
    print(f"  mean  = {mean(label_1):.4f}")
    print(f"  std   = {std(label_1):.4f}")
    print(f"  min   = {min(label_1):.4f}")
    print(f"  p50   = {pct(label_1, 50):.4f}")
    print(f"  max   = {max(label_1):.4f}")

    # Low similarity vulnerable samples
    vuln_with_sim = [(s, samples[i]) for i, s in enumerate(sims) if samples[i].label == "1"]
    low_vuln = sorted(vuln_with_sim, key=lambda x: x[0])[:50]

    print("\n" + "=" * 60)
    print("LOWEST SIMILARITY VULNERABLE SAMPLES (label=1, top 30)")
    print("=" * 60)
    for sim, ex in low_vuln[:30]:
        reason = ex.description[:100]
        cwe = ex.cwe_id
        func = ex.func[:80]
        print(f"\n  sim={sim:.4f}  cwe_id={cwe}")
        print(f"  reason: {reason}")
        print(f"  func:   {func}")

    # Low similarity safe samples
    safe_with_sim = [(s, samples[i]) for i, s in enumerate(sims) if samples[i].label == "0"]
    low_safe = sorted(safe_with_sim, key=lambda x: x[0])[:20]

    print("\n" + "=" * 60)
    print("LOWEST SIMILARITY SAFE SAMPLES (label=0, top 20)")
    print("=" * 60)
    for sim, ex in low_safe[:20]:
        reason = ex.description[:100]
        cwe = ex.cwe_id
        func = ex.func[:80]
        print(f"\n  sim={sim:.4f}  cwe_id={cwe}")
        print(f"  reason: {reason}")
        print(f"  func:   {func}")

    # Distribution bins
    bins = [(0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5),
            (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01)]
    print("\n" + "=" * 60)
    print("SIMILARITY DISTRIBUTION (all samples)")
    print("=" * 60)
    for lo, hi in bins:
        count = sum(1 for s in sims if lo <= s < hi)
        bar = "#" * int(60 * count / len(sims))
        print(f"  [{lo:.1f},{hi:.1f})  {count:6,}  {100*count/len(sims):5.1f}%  {bar}")

    print("\n" + "=" * 60)
    print("SIMILARITY DISTRIBUTION (vulnerable only)")
    print("=" * 60)
    for lo, hi in bins:
        count = sum(1 for s in label_1 if lo <= s < hi)
        bar = "#" * int(60 * count / max(len(label_1), 1))
        print(f"  [{lo:.1f},{hi:.1f})  {count:6,}  {100*count/max(len(label_1),1):5.1f}%  {bar}")

    print("\n" + "=" * 60)
    print(f"Analysis complete. Total time: {time.time() - t0:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
