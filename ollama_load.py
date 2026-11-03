
import subprocess, time, requests, json, os, re
from typing import Any, Dict

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

start_ollama()