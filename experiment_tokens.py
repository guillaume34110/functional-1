"""Train the corrected visual representation using the existing representative data."""
import subprocess
import sys
from pathlib import Path


def main():
    subprocess.run([sys.executable, "-m", "unittest", "-v", "test_model", "test_collect", "test_train", "test_token_model"], check=True)
    if not Path("data/manifest.json").exists():
        subprocess.run([sys.executable, "reserve_test.py"], check=True)
    subprocess.run([sys.executable, "train_tokens.py", "--output", "experiment-tokens", "--seconds", "2400", "--updates", "6000", "--threads", "2"], check=True)


if __name__ == "__main__":
    main()
