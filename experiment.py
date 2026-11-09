"""One bounded CPU training run; preserve the whole output folder on failure."""
import json
from pathlib import Path
import subprocess
import sys


def main():
    subprocess.run([sys.executable, "-m", "unittest", "-v", "test_model", "test_collect", "test_train"], check=True)
    output = Path("experiment")
    output.mkdir(exist_ok=False)
    subprocess.run([sys.executable, "collect.py", "--output", str(output / "dataset"), "--episodes", "64",
                    "--steps", "1400", "--seed", "1042026"], check=True)
    manifest = json.loads((output / "dataset/manifest.json").read_text())
    if manifest["coverage"]["lines_cleared"] == 0:
        raise RuntimeError("no real line clears captured: fix representative coverage before training")
    subprocess.run([sys.executable, "train.py", "--data", str(output / "dataset"), "--output", str(output / "training"),
                    "--updates", "6000", "--seconds", "2400", "--threads", "2"], check=True)
    subprocess.run([sys.executable, "evaluate.py", "--data", str(output / "dataset"), "--weights", str(output / "training/model.pt"),
                    "--output", str(output / "evaluation"), "--threads", "2"], check=True)


if __name__ == "__main__":
    main()
