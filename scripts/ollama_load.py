
"""Start a local Ollama server (if one isn't already running) and wait for it.

Handy on a fresh notebook/VM before a real-model run:  python scripts/ollama_load.py
Server output is written to /tmp/ollama.log.
"""

import subprocess
import time

import requests

def start_ollama():
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=5)
        if r.status_code == 200:
            print("Ollama already running.")
            print(r.text[:500])
            return
    except Exception:
        pass

    print("Starting Ollama...")
    log = open("/tmp/ollama.log", "w")
    subprocess.Popen(["ollama", "serve"], stdout=log, stderr=subprocess.STDOUT)

    for i in range(30):
        try:
            r = requests.get("http://localhost:11434/api/tags", timeout=5)
            if r.status_code == 200:
                print("Ollama running.")
                print(r.text[:500])
                return
        except Exception:
            time.sleep(1)

    raise RuntimeError("Ollama did not start. Check /tmp/ollama.log")


if __name__ == "__main__":
    start_ollama()
