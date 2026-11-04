#!/usr/bin/env bash
# Usage: tests/repro-issue-14.sh API MODEL. Exit 0: the problem is gone. 1: it shows. 2: the script could not judge.
# Issue #14: DSML closing tags reach a reply's content with no opener, finish "stop", no tool call.
# One way there: a DSML call whose string value holds the block's closer (`</｜DSML｜ calls>`, as in a file about DSML).
# The parser cut the block at that first closer, dropped the call, and a streamed reply showed only the closers left
# after it. The reply's form is forced with guided_regex (structured output, patch 0036), greedy, thinking off, so the
# run is deterministic; the parser reads the forced text exactly as it would read a sampled one. A control call with a
# plain value must parse in both modes, or the script cannot judge.
API=$1; MODEL=$2
if [ -z "$API" ] || [ -z "$MODEL" ]; then echo "usage: $0 API MODEL" >&2; exit 2; fi
exec python3 - "$API" "$MODEL" <<'PY'
import json, re, sys, urllib.error, urllib.request
api, model = sys.argv[1].rstrip("/"), sys.argv[2]
D = "｜DSML｜"
CLOSER = "</" + D
TOOLS = [{"type": "function", "function": {"name": "write", "description": "Write a file",
          "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                         "required": ["path", "content"]}}}]

def block(value):
    return (f"<{D} calls>\n<{D} invoke name=\"write\">\n<{D} parameter name=\"path\" string=\"true\">notes.md</{D} parameter>\n"
            f"<{D} parameter name=\"content\" string=\"true\">{value}</{D} parameter>\n</{D} invoke>\n</{D} calls>")

def judge_fail(msg):
    print(f"RESULT: cannot judge: {msg}"); sys.exit(2)

def ask(text, stream):
    body = {"model": model, "temperature": 0, "max_tokens": 256, "tools": TOOLS, "guided_regex": re.escape(text),
            "messages": [{"role": "user", "content": "Write notes.md with the write tool."}],
            "chat_template_kwargs": {"thinking": False, "enable_thinking": False}}
    if stream: body["stream"] = True
    req = urllib.request.Request(api + "/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            if not stream:
                o = json.load(r); c = o["choices"][0]
                calls = [(t["function"]["name"], json.loads(t["function"]["arguments"])) for t in c["message"].get("tool_calls") or []]
                return dict(content=c["message"].get("content") or "", calls=calls, finish=c.get("finish_reason"))
            content, finish, parts, done = "", None, {}, False
            for line in r:
                if not line.startswith(b"data: "): continue
                if line.strip() == b"data: [DONE]": done = True; break
                for c in json.loads(line[6:]).get("choices", []):
                    d = c.get("delta", {}); content += d.get("content") or ""; finish = c.get("finish_reason") or finish
                    for t in d.get("tool_calls") or []:
                        p = parts.setdefault(t.get("index", 0), {"name": "", "arguments": ""})
                        f = t.get("function") or {}
                        p["name"] += f.get("name") or ""; p["arguments"] += f.get("arguments") or ""
            if not done: judge_fail("stream ended without [DONE]")
            calls = [(p["name"], json.loads(p["arguments"] or "{}")) for _, p in sorted(parts.items())]
            return dict(content=content, calls=calls, finish=finish)
    except urllib.error.HTTPError as e:
        judge_fail(f"HTTP {e.code}: {e.read()[:200].decode(errors='ignore')}")
    except (urllib.error.URLError, OSError, ValueError, KeyError, IndexError) as e:
        judge_fail(f"{type(e).__name__}: {e}"[:200])

def good(r, value):
    return (r["calls"] == [("write", {"path": "notes.md", "content": value})] and r["finish"] == "tool_calls"
            and CLOSER not in r["content"])

def show(name, r):
    print(f"{name}: finish={r['finish']} tool_calls={json.dumps(r['calls'], ensure_ascii=False)} content={json.dumps(r['content'], ensure_ascii=False)}")

plain = "plain text"
for stream in (False, True):
    r = ask(block(plain), stream); show(f"control, {'stream' if stream else 'whole'}", r)
    if not good(r, plain): judge_fail("a plain DSML call does not parse; the server cannot judge this case")
held = f"end of block: </{D} calls>"
ok = True
for stream in (False, True):
    r = ask(block(held), stream); name = f"value holds the closer, {'stream' if stream else 'whole'}"; show(name, r)
    if not good(r, held):
        ok = False
        print(f"  FAIL {name}: expected one write call with the value intact and no DSML closer in content")
print("RESULT: " + ("fixed" if ok else "problem shows")); sys.exit(0 if ok else 1)
PY
