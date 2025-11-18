import subprocess
import sys


def download_spacy_model():
    try:
        subprocess.check_call(
            [sys.executable, "-m", "spacy", "download", "en_core_web_sm"]
        )
    except subprocess.CalledProcessError as e:
        print(f"Error downloading SpaCy model: {e}")
        sys.exit(1)


if __name__ == "__main__":
    download_spacy_model()
