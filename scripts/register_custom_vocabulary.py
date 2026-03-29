"""
Register custom vocabulary with Amazon Transcribe.

Prerequisites:
    pip install boto3
    AWS credentials configured (aws configure or env vars)

Usage:
    python scripts/register_custom_vocabulary.py
"""

import boto3
import os
import sys

VOCABULARY_NAME = "medical-terms-ja"
LANGUAGE_CODE = "ja-JP"
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
VOCAB_FILE = os.path.join(DATA_DIR, "custom_vocabulary.txt")


def main():
    if not os.path.exists(VOCAB_FILE):
        print(f"Error: {VOCAB_FILE} not found. Run prepare_custom_vocabulary.py first.")
        sys.exit(1)

    # Read vocabulary file
    with open(VOCAB_FILE, "r", encoding="utf-8") as f:
        vocab_content = f.read()

    # Count entries (minus header)
    entry_count = len(vocab_content.strip().split("\n")) - 1
    print(f"Registering {entry_count} terms as '{VOCABULARY_NAME}'...")

    client = boto3.client("transcribe")

    # Check if vocabulary already exists
    try:
        existing = client.get_vocabulary(VocabularyName=VOCABULARY_NAME)
        status = existing["VocabularyState"]
        print(f"Vocabulary '{VOCABULARY_NAME}' already exists (status: {status})")
        print("Updating...")
        response = client.update_vocabulary(
            VocabularyName=VOCABULARY_NAME,
            LanguageCode=LANGUAGE_CODE,
            VocabularyFileUri=None,  # Will use Phrases instead
        )
    except client.exceptions.BadRequestException:
        pass

    # Upload via S3 or use inline phrases
    # For large vocabularies, S3 upload is recommended
    # Here we use the file-based approach with S3

    # First, try creating/updating with the file content directly
    # Amazon Transcribe accepts vocabulary via S3 URI or inline
    # For 38K+ terms, S3 is required

    print("\nNote: For 38,000+ terms, you need to:")
    print("1. Upload data/custom_vocabulary.txt to an S3 bucket")
    print("2. Run this script with the S3 URI")
    print("")
    print("Example:")
    print("  aws s3 cp data/custom_vocabulary.txt s3://your-bucket/custom_vocabulary.txt")
    print("  Then update this script with the S3 URI")
    print("")

    s3_uri = os.environ.get("VOCAB_S3_URI")
    if not s3_uri:
        print("Set VOCAB_S3_URI environment variable to the S3 URI of the vocabulary file.")
        print("  export VOCAB_S3_URI=s3://your-bucket/custom_vocabulary.txt")
        print("  python scripts/register_custom_vocabulary.py")
        sys.exit(0)

    try:
        response = client.create_vocabulary(
            VocabularyName=VOCABULARY_NAME,
            LanguageCode=LANGUAGE_CODE,
            VocabularyFileUri=s3_uri,
        )
        print(f"Vocabulary creation started. Status: {response['VocabularyState']}")
        print("Check status with:")
        print(f"  aws transcribe get-vocabulary --vocabulary-name {VOCABULARY_NAME}")
    except client.exceptions.ConflictException:
        response = client.update_vocabulary(
            VocabularyName=VOCABULARY_NAME,
            LanguageCode=LANGUAGE_CODE,
            VocabularyFileUri=s3_uri,
        )
        print(f"Vocabulary update started. Status: {response['VocabularyState']}")


if __name__ == "__main__":
    main()
