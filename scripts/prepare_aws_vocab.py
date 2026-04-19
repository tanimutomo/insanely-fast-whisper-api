"""
Generate a compact Amazon Transcribe custom vocabulary (<=50KB) for ja-JP.

Amazon Transcribe limits custom vocabulary files to 50 KB. Characters are
constrained to the ja-JP character set (see data/ja_charset.txt, derived
from https://docs.aws.amazon.com/transcribe/latest/dg/samples/ja-jp-character-set.zip).
Notably no digits, no hyphen/period, and a specific subset of ASCII letters.

Priority: abbreviations > clinical terms > drugs > disease names.
Output: data/custom_vocabulary_aws.txt (one phrase per line).
"""

import csv
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
CHARSET = os.path.join(DATA_DIR, "ja_charset.txt")
OUTPUT = os.path.join(DATA_DIR, "custom_vocabulary_aws.txt")
MAX_BYTES = 49_000  # leave headroom under the 50 KB limit


def load_charset():
    chars = set()
    with open(CHARSET, "r", encoding="utf-8") as f:
        for line in f:
            c = line.rstrip("\n")
            if c:
                chars.add(c)
    return chars


def load(path, key, allowed):
    terms = []
    with open(os.path.join(DATA_DIR, path), "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            raw = row[key].strip()
            for term in raw.replace("、", ",").replace("・", ",").split(","):
                t = term.strip()
                if not t or len(t) > 256 or len(t) < 2:
                    continue
                if not all(c in allowed for c in t):
                    continue
                terms.append(t)
    return terms


def main():
    allowed = load_charset()
    abbr = load("medical_abbreviations.csv", "abbreviation", allowed)
    terms = load("medical_terms_ja.csv", "term", allowed)
    drugs = load("drug_names.csv", "drug_name", allowed)
    diseases = load("icd10_diseases_ja.csv", "disease_name", allowed)

    prioritized = []
    seen = set()
    for group in (abbr, terms, drugs, diseases):
        for t in group:
            if t not in seen:
                seen.add(t)
                prioritized.append(t)

    total = 0
    kept = []
    for t in prioritized:
        size = len(t.encode("utf-8")) + 1
        if total + size > MAX_BYTES:
            break
        kept.append(t)
        total += size

    with open(OUTPUT, "w", encoding="utf-8") as f:
        f.write("\n".join(kept) + "\n")

    print(f"Wrote {len(kept)} terms, {total} bytes -> {OUTPUT}")


if __name__ == "__main__":
    main()
