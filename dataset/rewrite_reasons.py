"""
Load dataset/vcldata.pkl, apply FLAW-based reasoning enhancement,
overwrite the reason field, and save back to the same pickle.
"""

import pickle
import re
from collections import defaultdict
from tqdm import tqdm

FLAW_PATTERN = re.compile(
    r'/\*\s*(?:POTENTIAL\s+)?FLAW:?\s*([^*]+)\*/', re.IGNORECASE
)


def is_safe_cwe(cwe_id):
    return cwe_id is None or str(cwe_id) == "None"


def is_generic_reason(reason):
    if not reason:
        return True
    return reason in (
        "This function is a secure function.",
        "This function may cause a None.",
        "This function may cause a Code Quality.",
    )


def extract_flaw_text(func):
    matches = FLAW_PATTERN.findall(func or "")
    if matches:
        return matches[0].strip()
    return None


def extract_operation_template(reason):
    m = re.match(r"This function first (.+?) and then (.+)", reason)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return None, None


def build_cwe_templates(dataset):
    cwe_templates = defaultdict(list)
    for sample in dataset:
        if is_safe_cwe(sample.get("cwe_id")):
            continue
        reason = sample.get("reason", "")
        op, effect = extract_operation_template(reason)
        if op:
            cwe_templates[sample.get("cwe_id")].append((op, effect))
    return cwe_templates


def rewrite_reason(sample, cwe_templates, cwe_names):
    cwe_id = sample.get("cwe_id", "None")
    func = sample.get("func", "")
    reason = sample.get("reason", "")
    label = str(sample.get("label", "0"))

    is_safe = label == "0"

    if not is_safe:
        flaw = extract_flaw_text(func)
        if flaw:
            return flaw

    op, _ = extract_operation_template(reason)
    if op:
        return reason

    if is_safe:
        return "This is a secure function with no vulnerability."

    if is_generic_reason(reason):
        cwe_str = str(cwe_id)
        templates = cwe_templates.get(cwe_str, [])
        if templates:
            best_op, _ = templates[0]
            cwe_name = cwe_names.get(cwe_str, f"CWE-{cwe_str}")
            return f"{cwe_name}: {best_op}."

        cwe_name = cwe_names.get(cwe_str, f"CWE-{cwe_str}")
        return f"A vulnerability of {cwe_name}."

    return reason


def main():
    import json

    print("Loading dataset/vcldata.pkl...")
    with open("dataset/vcldata.pkl", "rb") as f:
        raw = pickle.load(f)
    print(f"  Total samples: {len(raw)}")

    cwe_names = json.load(open("src/cwe_names.json"))

    print("Building CWE operation templates...")
    cwe_templates = build_cwe_templates(raw)
    print(f"  CWEs with templates: {len(cwe_templates)}")

    print("Rewriting reasons...")
    for sample in tqdm(raw):
        sample["reason"] = rewrite_reason(sample, cwe_templates, cwe_names)

    print("Saving back to dataset/vcldata.pkl...")
    with open("dataset/vcldata.pkl", "wb") as f:
        pickle.dump(raw, f)
    print("  Done.")


if __name__ == "__main__":
    main()
