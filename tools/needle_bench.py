"""Needle-in-a-haystack retrieval sweep: a random code hidden at a chosen depth inside a filler prompt of a chosen
length, across a grid of (length, depth) pairs, checking the server's greedy reply names it.

    python3 tools/needle_bench.py [--url http://127.0.0.1:8888] [--model NAME] [--lens 4096,8192,...]
        [--depths 0,25,50,75,100] [--reps 1] [--json OUT.jsonl]

Needs only the Python standard library. Builds each length's haystack from a fixed filler sentence, calibrated once
against the live server (one short request) to turn "tokens" into "sentence repeats" without needing a local
tokenizer -- so the actual prompt_tokens for a given --lens is approximate, not exact. The needle ("The secret code
is: NNNNNN.") is inserted at the chosen percentage of the haystack; the question ("What is the secret code? Reply
with the number only.") is appended after it, thinking off, greedy. A cell is a PASS if the reply contains the
exact code.
"""
import json
import random
import sys
import time
import urllib.error
import urllib.request

args = sys.argv[1:]


def arg(name, default):
    return args[args.index(name) + 1] if name in args else default


URL = arg("--url", "http://127.0.0.1:8888").rstrip("/")
MODEL = arg("--model", "DeepSeek-V4.1-Flash-1017")
LENS = [int(x) for x in arg("--lens", "4096,8192,16384,32768,65536,131072,262144,524288,1000000").split(",")]
DEPTHS = [int(x) for x in arg("--depths", "0,25,50,75,100").split(",")]
REPS = int(arg("--reps", "1"))
JSON_OUT = arg("--json", None)

FILLER = "The quick brown fox jumps over the lazy dog near the riverbank at dawn. "


def post(body, timeout=1800):
    req = urllib.request.Request(URL + "/v1/chat/completions", json.dumps(body).encode(), {"content-type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        return {"error": e.read().decode(errors="replace")}


def chat(prompt, max_tokens=20):
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
            "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
    return post(body)


def text(r):
    """The reply's text when r is a completion with non-empty text content, else None: an HTTP error, an error object,
    no choices or empty content is never read as the model's reply."""
    try:
        c = r["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return None
    return c if isinstance(c, str) and c and "error" not in r else None


def problem(r):
    """What went wrong with a reply text() refused, for the log."""
    return "error: " + json.dumps(r)[:200] if isinstance(r, dict) and "error" in r else "no reply text: " + json.dumps(r)[:200]


def prompt_tokens(r):
    return (r.get("usage") or {}).get("prompt_tokens")


calib_repeats = 200
calib = chat(FILLER * calib_repeats + "\nSay OK.", max_tokens=5)
calib_tok = prompt_tokens(calib)
if not calib_tok:
    print("Calibration request failed:", calib)
    sys.exit(1)
tokens_per_repeat = calib_tok / calib_repeats
print(f"Calibration: {calib_tok} tokens for {calib_repeats} filler repeats ({tokens_per_repeat:.2f} tokens/repeat)\n", flush=True)

results = []
for length in LENS:
    repeats = max(1, int(length / tokens_per_repeat))
    haystack = [FILLER] * repeats
    for depth in DEPTHS:
        pos = min(int(repeats * depth / 100), repeats)
        for rep in range(REPS):
            code = str(random.randint(100000, 999999))
            needle = f"The secret code is: {code}. "
            prompt = "".join(haystack[:pos]) + needle + "".join(haystack[pos:]) + "\nWhat is the secret code? Reply with the number only."
            t0 = time.time()
            r = chat(prompt, max_tokens=20)
            dt = time.time() - t0
            tok = prompt_tokens(r)
            reply = text(r)
            ok = reply is not None and code in reply
            row = {"length": length, "depth": depth, "rep": rep, "prompt_tokens": tok, "ok": ok, "reply": (reply or "")[:40], "seconds": round(dt, 1)}
            if reply is None:
                row["error"] = problem(r)
            results.append(row)
            shown = f"reply={reply[:30]!r}" if reply is not None else row["error"]
            print(f"len={length:>8} depth={depth:>3}% tok={tok} ok={ok} {dt:>6.1f}s {shown}", flush=True)
            if JSON_OUT:
                with open(JSON_OUT, "a") as f:
                    f.write(json.dumps(row) + "\n")

passed = sum(1 for r in results if r["ok"])
print(f"\n{passed}/{len(results)} passed\n")
grid = {}
for r in results:
    grid.setdefault(r["length"], {})[r["depth"]] = "PASS" if r["ok"] else "FAIL"
print("length      " + "  ".join(f"{d:>4}%" for d in DEPTHS))
for length in LENS:
    print(f"{length:>10}  " + "  ".join(f"{grid.get(length, {}).get(d, '?'):>5}" for d in DEPTHS))
sys.exit(0 if passed == len(results) else 1)
