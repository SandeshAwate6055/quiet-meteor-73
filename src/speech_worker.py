"""Optional pyttsx3 worker, isolated so a broken driver can be timed out."""
import sys
from pathlib import Path


def main():
    import pyttsx3
    engine = pyttsx3.init()
    engine.setProperty("rate", 150)
    engine.save_to_file(Path(sys.argv[1]).read_text(encoding="utf-8"), sys.argv[2])
    engine.runAndWait()
    engine.stop()


if __name__ == "__main__":
    main()
