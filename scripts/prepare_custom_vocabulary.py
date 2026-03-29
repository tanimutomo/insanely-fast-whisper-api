"""
Prepare custom vocabulary for Amazon Transcribe from medical term CSVs.

Amazon Transcribe custom vocabulary format (TSV):
Phrase\tIPA\tSoundsLike\tDisplayAs

For Japanese, we use:
- Phrase: the term (how it should be recognized)
- SoundsLike: reading in katakana (optional)
- DisplayAs: how it should appear in output (optional)

Usage:
    python scripts/prepare_custom_vocabulary.py

Output:
    data/custom_vocabulary.txt (TSV format for Amazon Transcribe)
"""

import csv
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
OUTPUT = os.path.join(DATA_DIR, "custom_vocabulary.txt")


def load_diseases():
    """Load ICD-10 disease names."""
    terms = []
    path = os.path.join(DATA_DIR, "icd10_diseases_ja.csv")
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["disease_name"].strip()
            kana = row["disease_kana"].strip()
            if name and len(name) >= 2:
                terms.append({
                    "phrase": name,
                    "sounds_like": kana if kana else "",
                    "display_as": name,
                })
    return terms


def load_drugs():
    """Load drug names."""
    terms = []
    path = os.path.join(DATA_DIR, "drug_names.csv")
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["drug_name"].strip()
            if name and len(name) >= 2:
                terms.append({
                    "phrase": name,
                    "sounds_like": "",
                    "display_as": name,
                })
    return terms


def load_medical_terms():
    """Load medical terms (anatomy, symptoms, etc.)."""
    terms = []
    path = os.path.join(DATA_DIR, "medical_terms_ja.csv")
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            term = row["term"].strip()
            if term and len(term) >= 2:
                terms.append({
                    "phrase": term,
                    "sounds_like": "",
                    "display_as": term,
                })
    return terms


def load_abbreviations():
    """Load medical abbreviations."""
    terms = []
    path = os.path.join(DATA_DIR, "medical_abbreviations.csv")
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            abbr = row["abbreviation"].strip()
            japanese = row.get("japanese", "").strip()
            if abbr and len(abbr) >= 2:
                terms.append({
                    "phrase": abbr,
                    "sounds_like": japanese if japanese else "",
                    "display_as": abbr,
                })
    return terms


def main():
    all_terms = []
    print("Loading medical terms...")

    diseases = load_diseases()
    print(f"  Diseases: {len(diseases)}")
    all_terms.extend(diseases)

    drugs = load_drugs()
    print(f"  Drugs: {len(drugs)}")
    all_terms.extend(drugs)

    medical = load_medical_terms()
    print(f"  Medical terms: {len(medical)}")
    all_terms.extend(medical)

    abbreviations = load_abbreviations()
    print(f"  Abbreviations: {len(abbreviations)}")
    all_terms.extend(abbreviations)

    # Deduplicate by phrase
    seen = set()
    unique_terms = []
    for t in all_terms:
        if t["phrase"] not in seen:
            seen.add(t["phrase"])
            unique_terms.append(t)

    # Amazon Transcribe has a limit of 50,000 entries
    if len(unique_terms) > 50000:
        print(f"  Warning: {len(unique_terms)} terms exceeds 50,000 limit, truncating")
        unique_terms = unique_terms[:50000]

    # Write TSV
    with open(OUTPUT, "w", encoding="utf-8") as f:
        f.write("Phrase\tIPA\tSoundsLike\tDisplayAs\n")
        for t in unique_terms:
            phrase = t["phrase"]
            sounds_like = t["sounds_like"]
            display_as = t["display_as"]
            f.write(f"{phrase}\t\t{sounds_like}\t{display_as}\n")

    print(f"\nTotal unique terms: {len(unique_terms)}")
    print(f"Output: {OUTPUT}")


if __name__ == "__main__":
    main()
