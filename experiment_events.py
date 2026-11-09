"""Correct the measured erase-only failure on the same representative training data."""
import subprocess
import sys


def main():
    subprocess.run([sys.executable, "-m", "unittest", "-v", "test_model", "test_collect", "test_train", "test_token_model"], check=True)
    subprocess.run([sys.executable, "train_tokens.py", "--events", "--warm-start", "weights/tiles-v2.pt",
                    "--output", "experiment-events", "--seconds", "2400", "--updates", "6000", "--threads", "2"], check=True)
    subprocess.run([sys.executable, "diagnose.py", "--weights", "experiment-events/model.pt"], check=True)


if __name__ == "__main__":
    main()
