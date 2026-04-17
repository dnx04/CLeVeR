"""
Reason and CWE augmentation for vcldata.pkl via LLM.

Rules:
  1. Unlabeled vuln (cwe_id="None", label=1) → LLM generates BOTH reason AND CWE ID
  2. Labeled vuln (specific CWE, label=1, sim < threshold) → LLM augments reason only
  3. Safe (label=0) → untouched

Usage:
    uv run python dataset/augment_dataset.py [--sim-threshold 0.4] [--dry-run] [--no-llm]
"""

import pickle
import re
import os
import time
import json
import argparse
import sys

import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

load_dotenv(".env.local")

API_KEY = os.environ.get("FPTAI_API_KEY", "")
BASE_URL = "https://mkp-api.fptcloud.com/v1"
MODEL = "gemma-4-31B-it"
MAX_TOKENS = 64
MAX_RETRIES = 3
CONCURRENCY = 16
SIM_THRESHOLD = 0.4

_src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)


# -------------------------------------------------------------------------
# UniXcoder
# -------------------------------------------------------------------------

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
    model, tokenizer, device = get_model()
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
        batch = texts[i:i + batch_size]
        tokens = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length
        ).to(device)
        with torch.no_grad():
            emb = model(**tokens).last_hidden_state[:, 0, :]
        emb = emb / emb.norm(dim=-1, keepdim=True)
        embeddings.append(emb.cpu())
    return torch.cat(embeddings, dim=0)


def get_field(ex, name, default=""):
    return getattr(ex, name, ex.get(name, default) if hasattr(ex, "get") else default)


# -------------------------------------------------------------------------
# LLM helpers
# -------------------------------------------------------------------------

def build_unlabeled_prompt(code_snippet):
    return (
        "You are a vulnerability analysis assistant. Analyze the following code snippet "
        "from an unlabeled vulnerability sample (vulnerable but no specific CWE type identified). "
        "Generate a SINGLE concise sentence describing the specific vulnerability mechanism. "
        "IMPORTANT: Your output MUST include the correct CWE ID (e.g. CWE-78, CWE-121). "
        "Focus on the concrete unsafe operation observed.\n\n"
        f"Code:\n{code_snippet}\n\n"
        "Output ONLY ONE sentence that includes the CWE ID:"
    )


def build_labeled_prompt(cwe_id, cwe_name, code_snippet, source, sink):
    return (
        "You are a vulnerability analysis assistant. Given the following code snippet, "
        "generate a SINGLE concise sentence describing the specific vulnerability mechanism. "
        "Focus on the concrete unsafe operation.\n\n"
        f"CWE-{cwe_id} ({cwe_name}):\n"
        f"Code:\n{code_snippet}\n"
        f"Source: {source}\n"
        f"Sink: {sink}\n\n"
        "Output ONLY ONE sentence:"
    )


def call_gemma(client, prompt, max_retries=MAX_RETRIES):
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=1.0,
                max_tokens=MAX_TOKENS,
            )
            return response.choices[0].message.content.strip()
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
    return None


CWE_ID_PATTERN = re.compile(r'CWE-(\d+)', re.IGNORECASE)


def clean_result(result):
    if not result:
        return None
    clean = result.strip().strip('"').strip("'").strip(".")
    if not clean.endswith("."):
        clean += "."
    return clean


def extract_cwe_id(text):
    m = CWE_ID_PATTERN.search(text)
    return m.group(1) if m else None


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Augment vcldata.pkl via LLM")
    parser.add_argument("--sim-threshold", type=float, default=SIM_THRESHOLD,
                        help=f"Similarity threshold for labeled vuln (default: {SIM_THRESHOLD})")
    parser.add_argument("--dry-run", action="store_true", help="Print changes without saving")
    parser.add_argument("--no-llm", action="store_true", help="Skip LLM rewriting")
    parser.add_argument("--batch-size", type=int, default=64, help="UniXcoder batch size")
    parser.add_argument("--concurrency", type=int, default=CONCURRENCY,
                        help=f"LLM concurrent calls (default: {CONCURRENCY})")
    args = parser.parse_args()

    t0 = time.time()

    print("Loading dataset/vcldata.pkl...")
    with open("dataset/vcldata.pkl", "rb") as f:
        data = pickle.load(f)
    print(f"  Total samples: {len(data)}\n")

    cwe_names = json.load(open("src/cwe_names.json"))

    # -------------------------------------------------------------------------
    # Categorize samples
    # -------------------------------------------------------------------------
    unlabeled_vuln = []
    labeled_vuln_below = []
    labeled_vuln_above = []

    # Compute similarities for labeled vuln samples
    print("Computing similarities for labeled vuln samples...")
    labeled_indices = []
    labeled_codes = []
    labeled_reasons = []
    for idx, ex in enumerate(data):
        cwe_id = str(get_field(ex, "cwe_id", "None"))
        label = str(get_field(ex, "label", "0"))
        if label == "1" and cwe_id != "None":
            labeled_indices.append(idx)
            labeled_codes.append(get_field(ex, "func", ""))
            labeled_reasons.append(get_field(ex, "description", ""))

    if labeled_codes:
        code_embs = encode_batch(labeled_codes, batch_size=args.batch_size)
        desc_embs = encode_batch(labeled_reasons, batch_size=args.batch_size)
        sims = ((code_embs * desc_embs).sum(dim=-1)).tolist()

        for idx, sim in zip(labeled_indices, sims):
            if sim < args.sim_threshold:
                labeled_vuln_below.append(idx)
            else:
                labeled_vuln_above.append(idx)

    # Unlabeled vuln
    for idx, ex in enumerate(data):
        cwe_id = str(get_field(ex, "cwe_id", "None"))
        label = str(get_field(ex, "label", "0"))
        if label == "1" and cwe_id == "None":
            unlabeled_vuln.append(idx)

    print(f"  Unlabeled vuln (label=1, cwe_id=None): {len(unlabeled_vuln)}")
    print(f"  Labeled vuln below threshold (< {args.sim_threshold}): {len(labeled_vuln_below)}")
    print(f"  Labeled vuln above threshold (>= {args.sim_threshold}): {len(labeled_vuln_above)}")
    print(f"  Safe (label=0): {sum(1 for ex in data if str(get_field(ex, 'label', '0')) == '0')}")

    to_process = unlabeled_vuln + labeled_vuln_below
    print(f"\nTotal samples to process via LLM: {len(to_process)}")

    llm_client = None
    if not args.no_llm and API_KEY:
        llm_client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
        print(f"LLM enabled (Gemma 4, {args.concurrency} workers)")
    elif not args.no_llm and not API_KEY:
        print("WARNING: FPTAI_API_KEY not set — skipping LLM")
    else:
        print("LLM disabled (--no-llm)")

    if args.no_llm or not llm_client or not to_process:
        elapsed = time.time() - t0
        print(f"\nDone in {elapsed:.1f}s")
        return

    # -------------------------------------------------------------------------
    # Build tasks
    # -------------------------------------------------------------------------
    all_tasks = []
    for idx in to_process:
        ex = data[idx]
        cwe_id = str(get_field(ex, "cwe_id", "None"))
        label = str(get_field(ex, "label", "0"))
        is_unlabeled = cwe_id == "None" and label == "1"
        reason = get_field(ex, "description", "")

        if is_unlabeled:
            cwe_name = "unlabeled"
            prompt = build_unlabeled_prompt(get_field(ex, "func", ""))
            task_type = "unlabeled"
        else:
            cwe_name = cwe_names.get(cwe_id, f"CWE-{cwe_id}")
            prompt = build_labeled_prompt(
                cwe_id, cwe_name,
                get_field(ex, "func", ""),
                get_field(ex, "source", ""),
                get_field(ex, "sink", ""),
            )
            task_type = "labeled"

        all_tasks.append((idx, task_type, cwe_id, cwe_name, reason, prompt))

    # -------------------------------------------------------------------------
    # LLM processing
    # -------------------------------------------------------------------------
    total_calls = 0
    success = 0
    fail = 0
    changes = []

    def process_task(task):
        idx, task_type, cwe_id, cwe_name, old_reason, prompt = task
        # NOTE: cwe_id is locally captured here to avoid closure over loop var
        orig_cwe_id = cwe_id
        client = llm_client
        result = call_gemma(client, prompt)
        if not result:
            return idx, task_type, old_reason, old_reason, orig_cwe_id, "failed"

        if task_type == "unlabeled":
            new_cwe = extract_cwe_id(result)
            if new_cwe:
                return idx, task_type, old_reason, clean_result(result), new_cwe, "cwe_and_reason"
            # retry with emphasis
            result2 = call_gemma(client, prompt + "\n\nYou MUST include a valid CWE ID.")
            if result2:
                new_cwe = extract_cwe_id(result2)
                if new_cwe:
                    return idx, task_type, old_reason, clean_result(result2), new_cwe, "cwe_and_reason"
            return idx, task_type, old_reason, old_reason, orig_cwe_id, "failed"
        else:
            return idx, task_type, old_reason, clean_result(result), orig_cwe_id, "reason_only"

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {executor.submit(process_task, t): t for t in all_tasks}
        pbar = tqdm(total=len(all_tasks), unit="sample", desc="LLM")

        for future in as_completed(futures):
            idx, task_type, old_reason, new_reason, new_cwe, status = future.result()
            total_calls += 1
            if status == "failed":
                fail += 1
            else:
                success += 1
                changes.append((idx, task_type, old_reason, new_reason, new_cwe))
                if not args.dry_run:
                    setattr(data[idx], "description", new_reason)
                    if new_cwe != cwe_id and task_type == "unlabeled":
                        setattr(data[idx], "cwe_id", new_cwe)
            pbar.update(1)
            if total_calls % 200 == 0:
                print(f"\n  ({total_calls} calls, rate={total_calls / (time.time() - t0):.1f}/sec)")
                time.sleep(5)
        pbar.close()

    # -------------------------------------------------------------------------
    # Report
    # -------------------------------------------------------------------------
    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"LLM: success={success}, failed={fail}")
    print(f"  unlabeled (reason+cwe): {sum(1 for c in changes if c[1] == 'unlabeled')}")
    print(f"  labeled (reason only):  {sum(1 for c in changes if c[1] == 'labeled')}")
    print(f"Total time: {elapsed:.1f}s")

    if args.dry_run:
        print("\nDRY RUN — no changes saved")
        print(f"\n{len(changes)} samples would be modified:")
        for idx, task_type, old_reason, new_reason, new_cwe in changes[:20]:
            cwe_change = f" (cwe: {get_field(data[idx], 'cwe_id', 'None')} -> {new_cwe})" if task_type == "unlabeled" else ""
            print(f"  [{task_type}]{cwe_change} \"{old_reason}\" -> \"{new_reason}\"")
        if len(changes) > 20:
            print(f"  ... and {len(changes) - 20} more")
    else:
        print("Saving to dataset/vcldata.pkl...")
        with open("dataset/vcldata.pkl", "wb") as f:
            pickle.dump(data, f)
        print("  Done.")


if __name__ == "__main__":
    main()