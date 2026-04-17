"""
Unified reason enhancement for vcldata.pkl.

Priority order for reason sources:
  1. FLAW comment in code           → use directly
  2. Non-generic existing reason    → keep original
  3. Generic safe (label=0)        → safe description template
  4. Generic vuln (label=1)        → LLM enhance via Gemma 4

Priority for FLAW markers (highest to lowest):
  1. FLAW:           (confirmed vuln)
  2. POSSIBLE FLAW:  (possible vuln)
  3. POTENTIAL FLAW: (potential issue)
  4. INCIDENTAL FLAW: (incidental/lesser issue)

FIX: and INCIDENTAL: are skipped (not vulnerability descriptions).

Samples flagged for LLM enhancement:
  - Specific CWE with generic "may cause a CWE-X" pattern
  - Unlabeled vuln (cwe_id="None") with generic "may cause a None" pattern

Usage:
    uv run python dataset/rewrite_reasons.py [--dry-run] [--no-llm]
"""

import pickle
import re
import os
import time
import json
import argparse
import sys
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

load_dotenv(".env.local")

# Ensure src/ is on path so pickle can find ExampleFeature class
_src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

API_KEY = os.environ.get("FPTAI_API_KEY", "")
BASE_URL = "https://mkp-api.fptcloud.com/v1"
MODEL = "gemma-4-31B-it"
MAX_TOKENS = 64
MAX_RETRIES = 3
CONCURRENCY = 16

# FLAW markers in priority order (highest to lowest)
FLAW_PATTERNS = [
    ("FLAW:",           re.compile(r'/\*\s*FLAW:\s*([^*]+)\*/', re.IGNORECASE)),
    ("POSSIBLE FLAW:",  re.compile(r'/\*\s*POSSIBLE\s+FLAW:\s*([^*]+)\*/', re.IGNORECASE)),
    ("POTENTIAL FLAW:", re.compile(r'/\*\s*POTENTIAL\s+FLAW:\s*([^*]+)\*/', re.IGNORECASE)),
    ("INCIDENTAL FLAW:", re.compile(r'/\*\s*INCIDENTAL\s+FLAW:\s*([^*]+)\*/', re.IGNORECASE)),
]

# -------------------------------------------------------------------------
# Pattern detection
# -------------------------------------------------------------------------

GENERIC_SAFE_REASONS = (
    "This function is a secure function.",
    "This function may cause a None.",
    "This function may cause a Code Quality.",
)


def get_field(ex, name, default=""):
    """Get a field from ExampleFeature (attribute) or dict-like object."""
    return getattr(ex, name, ex.get(name, default) if hasattr(ex, "get") else default)


def is_generic_safe(reason):
    return reason in GENERIC_SAFE_REASONS


def is_generic_vuln_specific(reason):
    """Generic 'may cause a CWE-X' pattern for specific CWE types."""
    if not reason:
        return False
    return reason.startswith("This function may cause a ")


def is_generic_vuln_unlabeled(reason):
    """Generic patterns for unlabeled vuln samples (cwe_id='None', label=1)."""
    if not reason:
        return False
    if "may cause a None" in reason and reason.startswith("This function first"):
        return True
    if reason == "A vulnerability of CWE-None.":
        return True
    return False


def is_generic_vuln(reason):
    return is_generic_vuln_specific(reason) or is_generic_vuln_unlabeled(reason)


# -------------------------------------------------------------------------
# FLAW extraction
# -------------------------------------------------------------------------

def extract_flaw(func):
    """Extract FLAW text using priority-ordered markers (FLAW > POSSIBLE > POTENTIAL > INCIDENTAL)."""
    func_text = func or ""
    for marker_name, pattern in FLAW_PATTERNS:
        matches = pattern.findall(func_text)
        if matches:
            return matches[0].strip()
    return None


# -------------------------------------------------------------------------
# LLM helpers
# -------------------------------------------------------------------------

def build_llm_prompt(cwe_id, cwe_name, code_snippet, source, sink, is_unlabeled):
    if is_unlabeled:
        return (
            "You are a vulnerability analysis assistant. Given the following code snippet from an "
            "unlabeled vulnerability sample (vulnerable but no specific CWE type identified), "
            "generate a SINGLE concise sentence describing the specific vulnerability mechanism. "
            "IMPORTANT: Your output MUST include the CWE ID (e.g. CWE-78, CWE-121) if you can identify it. "
            "Focus on the concrete unsafe operation observed in the code.\n\n"
            f"Code:\n{code_snippet}\n\n"
            "Output only the vulnerability description (ONE sentence, MUST include CWE ID if identifiable):"
        )
    return (
        "You are a vulnerability analysis assistant. Given the following code snippet, "
        "generate a SINGLE concise sentence describing the specific vulnerability mechanism. "
        "Focus on the concrete unsafe operation.\n\n"
        f"CWE-{cwe_id} ({cwe_name}):\n"
        f"Code:\n{code_snippet}\n"
        f"Source: {source}\n"
        f"Sink: {sink}\n\n"
        "Output only the vulnerability description (ONE sentence):"
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


CWE_ID_PATTERN = re.compile(r'CWE-\d+', re.IGNORECASE)


def clean_result(result):
    if not result:
        return None
    clean = result.strip().strip('"').strip("'").strip(".")
    if not clean.endswith("."):
        clean += "."
    return clean


def validate_unlabeled_output(result):
    """For unlabeled vuln, LLM output MUST contain a CWE ID."""
    if not result:
        return False
    return bool(CWE_ID_PATTERN.search(result))


# -------------------------------------------------------------------------
# Core decision logic
# -------------------------------------------------------------------------

def decide_reason(sample, cwe_names, llm_client=None):
    """
    Return (new_reason, source) for the sample.
    source: 'flaw' | 'original' | 'safe_template' | 'llm' | 'failed'
    """
    func = getattr(sample, "func", "")
    cwe_id = str(getattr(sample, "cwe_id", "None"))
    label = str(getattr(sample, "label", "0"))
    old_reason = getattr(sample, "description", "")
    is_safe = label == "0"
    is_unlabeled = cwe_id == "None" and not is_safe

    # 1. FLAW comment → use directly
    flaw = extract_flaw(func)
    if flaw:
        return flaw, "flaw"

    # 2. Safe samples with generic reason → safe template
    if is_safe:
        if is_generic_safe(old_reason):
            return "This is a secure function with no vulnerability.", "safe_template"
        return old_reason, "original"

    # 3. Vuln samples — keep non-generic reasons
    if not is_generic_vuln(old_reason):
        return old_reason, "original"

    # 4. Vuln samples with generic reasons → LLM enhance (if client provided)
    if llm_client is None:
        return old_reason, "failed"

    cwe_name = cwe_names.get(cwe_id, f"CWE-{cwe_id}") if cwe_id != "None" else "unlabeled"
    prompt = build_llm_prompt(
        cwe_id, cwe_name,
        func,
        getattr(sample, "source", ""),
        getattr(sample, "sink", ""),
        is_unlabeled,
    )
    result = call_gemma(llm_client, prompt)
    if result:
        # For unlabeled vuln, LLM output MUST contain a CWE ID
        if is_unlabeled and not validate_unlabeled_output(result):
            result = call_gemma(llm_client, prompt + "\n\nIMPORTANT: You MUST include a CWE ID (e.g. CWE-78) in your response.")
            if result and not validate_unlabeled_output(result):
                return old_reason, "failed"
        if result:
            return clean_result(result), "llm"
    return old_reason, "failed"


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Unified reason enhancement")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print changes without saving")
    parser.add_argument("--no-llm", action="store_true",
                        help="Skip LLM enhancement (FLAW + templates only)")
    parser.add_argument("--concurrency", type=int, default=CONCURRENCY,
                        help=f"LLM concurrent calls (default: {CONCURRENCY})")
    args = parser.parse_args()

    t0 = time.time()

    print("Loading dataset/vcldata.pkl...")
    with open("dataset/vcldata.pkl", "rb") as f:
        data = pickle.load(f)
    print(f"  Total samples: {len(data)}\n")

    cwe_names = json.load(open("src/cwe_names.json"))

    llm_client = None
    if not args.no_llm and API_KEY:
        llm_client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
        print("LLM enhancement enabled (Gemma 4)")
    elif not args.no_llm and not API_KEY:
        print("WARNING: FPTAI_API_KEY not set — skipping LLM enhancement")
    else:
        print("LLM enhancement disabled (--no-llm)")

    # -------------------------------------------------------------------------
    # Pass 1: FLAW + template rewriting (no LLM needed)
    # -------------------------------------------------------------------------
    print("\nPass 1: FLAW extraction + template rewriting...")
    stats = {"flaw": 0, "original": 0, "safe_template": 0, "llm": 0, "failed": 0}
    changes = []

    for ex in tqdm(data, desc="Processing"):
        old_reason = get_field(ex, "description", "")
        new_reason, source = decide_reason(ex, cwe_names, llm_client=None)
        stats[source] += 1
        if new_reason != old_reason:
            changes.append((old_reason, new_reason, source))
        setattr(ex, "_reason_src", source)
        if not args.dry_run:
            setattr(ex, "description", new_reason)

    print(f"\nPass 1 stats: { {k: v for k, v in stats.items() if v > 0} }")

    # -------------------------------------------------------------------------
    # Pass 2: LLM enhancement for remaining problematic samples
    # -------------------------------------------------------------------------
    problematic = [ex for ex in data if getattr(ex, "_reason_src", None) in ("llm", "failed")]
    if args.no_llm or not problematic:
        print("\nNo samples need LLM enhancement.")
    else:
        total_calls = 0
        llm_success = 0
        llm_fail = 0

        all_tasks = []
        for ex in problematic:
            cwe_id = str(get_field(ex, "cwe_id", "None"))
            is_unlabeled = cwe_id == "None" and int(get_field(ex, "label", "0")) == 1
            cwe_name = cwe_names.get(cwe_id, f"CWE-{cwe_id}") if cwe_id != "None" else "unlabeled"
            all_tasks.append((ex, cwe_name, is_unlabeled, llm_client))

        def process_task(task):
            ex, cwe_name, is_unlabeled, client = task
            old_reason = get_field(ex, "description", "")
            cwe_id = str(get_field(ex, "cwe_id", "None"))
            code = get_field(ex, "func", "")
            src = get_field(ex, "source", "")
            sink = get_field(ex, "sink", "")
            prompt = build_llm_prompt(cwe_id, cwe_name, code, src, sink, is_unlabeled)
            result = call_gemma(client, prompt)
            if result:
                return old_reason, clean_result(result), "llm"
            return old_reason, old_reason, "failed"

        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            futures = {executor.submit(process_task, t): t for t in all_tasks}
            pbar = tqdm(total=len(problematic), unit="sample", desc="LLM")

            for future in as_completed(futures):
                old_reason, new_reason, status = future.result()
                total_calls += 1
                if status == "llm":
                    llm_success += 1
                else:
                    llm_fail += 1

                # Update sample in-place
                for ex in data:
                    if get_field(ex, "func", "") == get_field(all_tasks[futures[future]][0], "func", ""):
                        setattr(ex, "description", new_reason)
                        setattr(ex, "_reason_src", "llm" if status == "llm" else "failed")
                        break

                pbar.update(1)
                if total_calls % 200 == 0:
                    print(f"\n  ({total_calls} calls, rate={total_calls / (time.time() - t0):.1f}/sec)")
                    time.sleep(5)
            pbar.close()

        print(f"\nLLM stats: success={llm_success}, failed={llm_fail}")

    # -------------------------------------------------------------------------
    # Clean helper fields
    # -------------------------------------------------------------------------
    for ex in data:
        if hasattr(ex, "_reason_src"):
            delattr(ex, "_reason_src")

    # -------------------------------------------------------------------------
    # Report
    # -------------------------------------------------------------------------
    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"Done in {elapsed:.1f}s")
    if args.dry_run:
        print("DRY RUN — no changes saved")
        print(f"\n{len(changes)} samples would be modified:")
        for old_reason, new_reason, src in changes[:20]:
            print(f"  [{src}] \"{old_reason}\" -> \"{new_reason}\"")
        if len(changes) > 20:
            print(f"  ... and {len(changes) - 20} more")
    else:
        print("Saving to dataset/vcldata.pkl...")
        with open("dataset/vcldata.pkl", "wb") as f:
            pickle.dump(data, f)
        print("  Done.")


if __name__ == "__main__":
    main()
