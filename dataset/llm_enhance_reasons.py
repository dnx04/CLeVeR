"""
LLM-enhanced reasoning for samples where FLAW + template rewriting was insufficient.
Uses Gemma 4 via OpenAI-compatible API (FPT Cloud).

Usage:
    uv run python dataset/llm_enhance_reasons.py 2>&1 | tee dataset/llm_enhance.log

Input:  dataset/vcldata.pkl (already FLAW-enriched)
Output: dataset/vcldata.pkl (further enriched with LLM reasoning)
"""

import pickle
import os
import time
import json
import re
import threading
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

load_dotenv(".env.local")

API_KEY = os.environ.get("FPTAI_API_KEY", "")
BASE_URL = "https://mkp-api.fptcloud.com/v1"
MODEL = "gemma-4-31B-it"
MAX_TOKENS = 128
MAX_RETRIES = 3
CONCURRENCY = 16

FLAW_PATTERN = re.compile(r'/\*\s*(?:POTENTIAL\s+)?FLAW:?\s*([^*]+)\*/', re.IGNORECASE)

# Thread-safe stdout lock to prevent interleaved lines
print_lock = threading.Lock()


def pprint(msg):
    """Thread-safe print to stdout (piped to log file via tee)."""
    with print_lock:
        print(msg, flush=True)


def extract_flaw(func):
    matches = FLAW_PATTERN.findall(func or "")
    if matches:
        return matches[0].strip()
    return None


def build_llm_prompt(cwe_id, cwe_name, code_snippet, source, sink):
    return (
        "You are a vulnerability analysis assistant. Given the following code snippet, "
        "generate a concise (1-2 sentence) technical description of the specific vulnerability mechanism. "
        "Focus on the concrete unsafe operation. Do NOT use generic phrases like 'may cause a'.\n\n"
        f"CWE-{cwe_id} ({cwe_name}):\n"
        f"Code:\n{code_snippet}\n"
        f"Source: {source}\n"
        f"Sink: {sink}\n\n"
        "Generate only the vulnerability description (1-2 sentences):"
    )


def is_problematic_reason(reason):
    """Check if reason still has generic 'may cause a' pattern (specific CWE types)."""
    if not reason:
        return False
    return reason.startswith("This function may cause a ")


def is_problematic_unlabeled(reason):
    """Check if reason has generic patterns for unlabeled vuln samples."""
    if not reason:
        return False
    # Pattern 1: "first X and then Y, which may cause a None"
    if "may cause a None" in reason and reason.startswith("This function first"):
        return True
    # Pattern 2: fallback "A vulnerability of CWE-None."
    if reason == "A vulnerability of CWE-None.":
        return True
    return False


def find_problematic_samples(data):
    problematic = []
    for ex in data:
        cwe = str(ex.get("cwe_id", "None"))
        label = int(ex.get("label", "0"))
        reason = ex.get("reason", "")

        if label == 0:
            continue

        if cwe == "None" and label == 1:
            # Unlabeled vuln: check for generic "first X and then Y, may cause a None"
            if is_problematic_unlabeled(reason):
                problematic.append(ex)
        elif is_problematic_reason(reason):
            # Specific CWE type: check for generic "may cause a CWE-X"
            problematic.append(ex)

    return problematic


def call_gemma(client, prompt, max_retries=3):
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


def clean_result(result):
    if not result:
        return None
    clean = result.strip().strip('"').strip("'").strip(".")
    if not clean.endswith("."):
        clean += "."
    return clean


def process_sample(args):
    ex, cwe_name, client = args
    old_reason = ex["reason"]
    cwe_id = str(ex.get("cwe_id", "None"))
    is_unlabeled = cwe_id == "None" and int(ex.get("label", 0)) == 1

    if is_unlabeled:
        # Try FLAW first for unlabeled vuln
        flaw = extract_flaw(ex.get("func", ""))
        if flaw:
            ex["reason"] = flaw
            ex["_reason_src"] = "flaw"
            return cwe_id, old_reason, flaw, "flaw_success"

        # No FLAW — use LLM without CWE context
        # For "A vulnerability of CWE-None." samples, source/sink are useless ("This function may cause a None.")
        # Use only the code snippet for vulnerability analysis
        if old_reason == "A vulnerability of CWE-None.":
            prompt = (
                "You are a vulnerability analysis assistant. Given the following code snippet from an "
                "unlabeled vulnerability sample (vulnerable but no specific CWE type identified), "
                "generate a concise (1-2 sentence) technical description of the specific vulnerability mechanism. "
                "Focus on the concrete unsafe operation observed in the code. "
                "Do NOT use generic phrases like 'may cause a' or 'vulnerability of CWE-None'.\n\n"
                f"Code:\n{ex.get('func', '')}\n\n"
                "Generate only the vulnerability description (1-2 sentences):"
            )
        else:
            prompt = (
                "You are a vulnerability analysis assistant. Given the following code snippet "
                "(from an unlabeled vulnerability sample — vulnerable but no specific CWE type), "
                "generate a concise (1-2 sentence) technical description of the specific vulnerability mechanism. "
                "Focus on the concrete unsafe operation. Do NOT use generic phrases like 'may cause a'.\n\n"
                f"Code:\n{ex.get('func', '')}\n"
                f"Source: {ex.get('source', '')}\n"
                f"Sink: {ex.get('sink', '')}\n\n"
                "Generate only the vulnerability description (1-2 sentences):"
            )
        result = call_gemma(client, prompt)
        if result:
            clean = clean_result(result)
            ex["reason"] = clean
            ex["_reason_src"] = "gemma"
            return cwe_id, old_reason, clean, "success"
        else:
            ex["_reason_src"] = "failed"
            return cwe_id, old_reason, None, "failed"
    else:
        # Specific CWE type: use LLM with CWE context
        prompt = build_llm_prompt(
            cwe_id,
            cwe_name,
            ex.get("func", ""),
            ex.get("source", ""),
            ex.get("sink", ""),
        )

        result = call_gemma(client, prompt)

        if result:
            clean = clean_result(result)
            ex["reason"] = clean
            ex["_reason_src"] = "gemma"
            status = "success"
        else:
            clean = None
            ex["_reason_src"] = "failed"
            status = "failed"

        return cwe_id, old_reason, clean, status


def main():
    t0 = time.time()

    print("Loading dataset/vcldata.pkl...", flush=True)
    with open("dataset/vcldata.pkl", "rb") as f:
        data = pickle.load(f)
    print(f"  Total samples: {len(data)}\n", flush=True)

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    cwe_names = json.load(open("src/cwe_names.json"))

    print("Finding problematic samples...", flush=True)
    problematic = find_problematic_samples(data)
    print(f"  Found {len(problematic)} samples needing LLM enhancement\n", flush=True)

    if not problematic:
        print("No samples need LLM enhancement.", flush=True)
        return

    # Categorize
    unlabeled = [ex for ex in problematic if str(ex.get("cwe_id", "None")) == "None" and int(ex.get("label", 0)) == 1]
    specific = [ex for ex in problematic if str(ex.get("cwe_id", "None")) != "None"]

    pprint(f"  Unlabeled vuln: {len(unlabeled)}")
    pprint(f"  Specific CWE:   {len(specific)}\n")

    print(
        f"Starting LLM enhancement: {len(problematic)} samples, "
        f"{CONCURRENCY} concurrent calls\n",
        flush=True,
    )

    total_calls = 0
    success_count = 0
    fail_count = 0
    flaw_count = 0

    all_tasks = []
    for ex in problematic:
        cwe_id = str(ex.get("cwe_id", "None"))
        if cwe_id == "None":
            cwe_name = "unlabeled"
        else:
            cwe_name = cwe_names.get(cwe_id, f"CWE-{cwe_id}")
        all_tasks.append((ex, cwe_name, client))

    with ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
        futures = {executor.submit(process_sample, args): args[0] for args in all_tasks}

        pbar = tqdm(total=len(problematic), unit="sample", desc="LLM enhancing")

        for future in as_completed(futures):
            cwe_id, old_reason, new_reason, status = future.result()
            total_calls += 1

            if status == "flaw_success":
                success_count += 1
                flaw_count += 1
            elif status == "success":
                success_count += 1
            else:
                fail_count += 1

            pbar.update(1)

            # Log in real-time: "old_reason" -> "new_reason"
            pprint(f"[{status}] CWE-{cwe_id} | \"{old_reason}\" -> \"{new_reason}\"")

            # Rate limit pause
            if total_calls % 200 == 0 and total_calls > 0:
                elapsed = time.time() - t0
                rate = total_calls / elapsed
                pprint(f"\n  (pause at {total_calls} calls, rate={rate:.1f} calls/sec)\n")
                time.sleep(5)

        pbar.close()

    # Clean helper fields
    for ex in data:
        ex.pop("_reason_src", None)

    elapsed = time.time() - t0
    rate = total_calls / elapsed if elapsed > 0 else 0
    pprint("\n=== SUMMARY ===")
    pprint(f"  Total LLM calls:   {total_calls}")
    pprint(f"  FLAW used:         {flaw_count} ({flaw_count/total_calls*100:.1f}%)")
    pprint(f"  LLM success:      {success_count - flaw_count} ({(success_count - flaw_count)/total_calls*100:.1f}%)")
    pprint(f"  Failed:           {fail_count} ({fail_count/total_calls*100:.1f}%)")
    pprint(f"  Elapsed:          {elapsed:.1f}s ({rate:.1f} calls/sec)")

    pprint("\nSaving to dataset/vcldata.pkl...")
    with open("dataset/vcldata.pkl", "wb") as f:
        pickle.dump(data, f)
    pprint("  Done.")


if __name__ == "__main__":
    main()