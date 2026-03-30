import json, time
from pathlib import Path
import requests
import statistics

API = "http://127.0.0.1:8000/chat/"
INP = Path("backend/eval/eval_100.jsonl")
OUT = Path("backend/eval/results_100.jsonl")


def normalize_answer(a: str) -> str:
    a = (a or "").strip()
    # remove special tokens if any
    a = a.replace("<|end|>", "").strip()
    return a


OUT.parent.mkdir(parents=True, exist_ok=True)

latencies = []
with INP.open("r", encoding="utf-8") as f_in, OUT.open("w", encoding="utf-8") as f_out:
    for line in f_in:
        ex = json.loads(line)
        # Step 1 (Option A): test with larger retrieval
        payload = {
            "message": ex["message"],
            "top_k": 30,
            "kg_k": 30,
        }
        t0 = time.time()
        r = requests.post(API, json=payload, timeout=120)
        dt = time.time() - t0
        latencies.append(dt)

        if r.status_code != 200:
            rec = {
                **ex,
                "status": r.status_code,
                "error": r.text,
                "latency_s": dt,
            }
        else:
            data = r.json()
            rec = {
                **ex,
                "status": 200,
                "answer": normalize_answer(data.get("answer", "")),
                "sources": data.get("sources", []),
                "trace": data.get("trace", {}),
                "latency_s": dt,
            }
        f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")

print("Saved:", OUT)
print("Latency mean:", round(statistics.mean(latencies), 3), "s",
      "p95:", round(sorted(latencies)[int(0.95*(len(latencies)-1))], 3), "s")


