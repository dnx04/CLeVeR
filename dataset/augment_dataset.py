"""
Reason and CWE augmentation for vcldata.pkl via LLM.

Only processes unlabeled vulnerable samples (cwe_id="None", label=1).
LLM generates BOTH reason AND CWE ID → updates description AND cwe_id.

Usage:
    uv run python dataset/augment_dataset.py [--dry-run] [--no-llm]
"""

import pickle
import re
import os
import time
import json
import argparse
import sys

from openai import OpenAI
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

load_dotenv(".env.local")

API_KEY = os.environ.get("FPTAI_API_KEY", "")
BASE_URL = "https://mkp-api.fptcloud.com/v1"
MODEL = "gemma-4-31B-it"
MAX_TOKENS = 128
MAX_RETRIES = 3
CONCURRENCY = 16


# -------------------------------------------------------------------------
# LLM helpers
# -------------------------------------------------------------------------

def build_unlabeled_prompt(code_snippet):
    return (
        "You are a vulnerability analysis assistant. Analyze the following code snippet "
        "(vulnerable but no specific CWE type identified). "
        "Identify the vulnerability SOURCE (where tainted data enters), "
        "the SINK (where unsafe operation occurs), and the MECHANISM (how the vulnerability occurs). "
        "Output format: \"CWE-ID: xxx | SOURCE: ... | SINK: ... | MECHANISM: <short sentence>\" "
        "Use numeric CWE-ID only (e.g. 78, 121, no CWE- prefix). "
        "SOURCE and SINK must be only variable/function names (e.g. getenv, strlen, data, buf). Be concise.\n\n"
        f"Code:\n{code_snippet}\n\n"
        "Output:"
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


CWE_ID_PATTERN = re.compile(r'(?:CWE[- ]ID[:\s]*)?(\d+)(?:\s+\|)', re.IGNORECASE)
SOURCE_PATTERN = re.compile(r'SOURCE:\s*([^|]+)', re.IGNORECASE)
SINK_PATTERN = re.compile(r'SINK:\s*([^|]+)', re.IGNORECASE)
MECHANISM_PATTERN = re.compile(r'MECHANISM:\s*(.+)', re.IGNORECASE)


def parse_llm_output(result):
    """Parse LLM output into (cwe_id, source, sink, mechanism)."""
    if not result:
        return None, None, None, None

    cwe_match = CWE_ID_PATTERN.search(result)
    source_match = SOURCE_PATTERN.search(result)
    sink_match = SINK_PATTERN.search(result)
    mech_match = MECHANISM_PATTERN.search(result)

    cwe_id = cwe_match.group(1) if cwe_match else None
    source = source_match.group(1).strip() if source_match else None
    sink = sink_match.group(1).strip() if sink_match else None
    mechanism = mech_match.group(1).strip().rstrip(".") if mech_match else None

    return cwe_id, source, sink, mechanism


def clean_result(result):
    if not result:
        return None
    return result.strip().strip('"').strip("'").strip(".")


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Augment vcldata.pkl via LLM")
    parser.add_argument("--dry-run", action="store_true", help="Print changes without saving")
    parser.add_argument("--no-llm", action="store_true", help="Skip LLM rewriting")
    parser.add_argument("--concurrency", type=int, default=CONCURRENCY,
                        help=f"LLM concurrent calls (default: {CONCURRENCY})")
    args = parser.parse_args()

    t0 = time.time()

    print("Loading dataset/vcldata.pkl...")
    with open("dataset/vcldata.pkl", "rb") as f:
        data = pickle.load(f)
    print(f"  Total samples: {len(data)}\n")

    # -------------------------------------------------------------------------
    # Find unlabeled vulnerable samples
    # -------------------------------------------------------------------------
    unlabeled_vuln = []
    for idx, ex in enumerate(data):
        cwe_id = str(getattr(ex, "cwe_id", "None"))
        label = str(getattr(ex, "label", "0"))
        if label == "1" and cwe_id == "None":
            unlabeled_vuln.append(idx)

    print(f"  Unlabeled vuln (label=1, cwe_id=None): {len(unlabeled_vuln)}")
    print(f"  Safe (label=0): {sum(1 for ex in data if str(getattr(ex, 'label', '0')) == '0')}")

    to_process = unlabeled_vuln
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
        func = getattr(ex, "func", "")
        prompt = build_unlabeled_prompt(func)
        all_tasks.append((idx, func, prompt))

    # -------------------------------------------------------------------------
    # LLM processing
    # -------------------------------------------------------------------------
    total_calls = 0
    success = 0
    fail = 0
    changes = []

    def process_task(task):
        idx, func, prompt = task
        client = llm_client
        result = call_gemma(client, prompt)
        if not result:
            return idx, None, None, None, None, "failed"

        cwe_id, source, sink, mechanism = parse_llm_output(result)
        if not cwe_id:
            # retry with emphasis
            result2 = call_gemma(client, prompt + "\n\nYou MUST include a valid CWE-ID.")
            if result2:
                cwe_id, source, sink, mechanism = parse_llm_output(result2)

        if cwe_id and source and sink and mechanism:
            description = f"SOURCE: {source} | SINK: {sink} | MECHANISM: {mechanism}."
            return idx, description, cwe_id, source, sink, "success"
        return idx, None, None, None, None, "failed"

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {executor.submit(process_task, t): t for t in all_tasks}
        pbar = tqdm(total=len(all_tasks), unit="sample", desc="LLM")

        for future in as_completed(futures):
            idx, new_reason, new_cwe, source, sink, status = future.result()
            total_calls += 1
            if status == "failed":
                fail += 1
            else:
                success += 1
                old_reason = getattr(data[idx], "description", "")
                old_cwe = getattr(data[idx], "cwe_id", "None")
                changes.append((idx, old_reason, new_reason, old_cwe, new_cwe))
                if not args.dry_run:
                    setattr(data[idx], "description", new_reason)
                    setattr(data[idx], "cwe_id", new_cwe)
                    setattr(data[idx], "source", source)
                    setattr(data[idx], "sink", sink)
            pbar.set_postfix(success=success, fail=fail)
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
    print(f"Total time: {elapsed:.1f}s")

    if args.dry_run:
        print("\nDRY RUN — no changes saved")
        print(f"\n{len(changes)} samples would be modified:")
        for idx, old_reason, new_reason, old_cwe, new_cwe in changes[:20]:
            print(f"  [{idx}] cwe: {old_cwe} -> {new_cwe}")
            print(f"       reason: \"{old_reason}\" -> \"{new_reason}\"")
        if len(changes) > 20:
            print(f"  ... and {len(changes) - 20} more")
    else:
        print("Saving to dataset/vcldata.pkl...")
        with open("dataset/vcldata.pkl", "wb") as f:
            pickle.dump(data, f)
        print("  Done.")


if __name__ == "__main__":
    main()