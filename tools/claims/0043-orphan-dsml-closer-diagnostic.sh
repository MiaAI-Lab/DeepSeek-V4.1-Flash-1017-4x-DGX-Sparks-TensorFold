#!/usr/bin/env bash
# Written by Hector Isaza (hecisaza, https://github.com/hecisaza) with the verify step of inference-agent-kit
# (https://github.com/hecisaza/inference-agent-kit). Apache-2.0. Unchanged below this header from
# https://github.com/hecisaza/DeepSeek-V4.1-Flash-1017-4x-DGX-Sparks-TensorFold/blob/a6395a89a08d1ca6464c32bcc5f4bd9dd8ce1cb6/claims/MiaAI-Lab-DeepSeek-V4.1-Flash-1017-4x-DGX-Sparks-TensorFold-pr-32.sh
# Usage: tools/claims/0043-orphan-dsml-closer-diagnostic.sh API MODEL [LOGCMD]. Exit 0: the claim holds. 1: it does not. 2: the script could not judge.
# claim: patches: 0043 a reply made only of orphan DSML closing tags is flagged (tool_parse_warning), not repaired.
# Tested: with tools offered, an answer made only of DSML closing tags (V4 and V4.1 spellings) gets
#   tensorfold.tool_parse_warning = "orphan_dsml_closers" in the whole reply and in the final SSE event; the content,
#   finish reason and usage are unchanged (content is exactly the closers, finish "stop", no tool_calls, same token_sha
#   and completion_tokens whole vs stream); the same closers without tools and closers after prose are not flagged;
#   when LOGCMD is given, one log line names the reply id and the code, without the reply text.
# How: the model declines to emit closers-only replies on request (PR.md), so the answer's form is forced with
#   guided_regex (structured output from 0036, already on the base); the diagnostic only reads the final content, so
#   the forced reply reaches the changed code exactly as a sampled one would. Greedy, thinking off, fixed inputs.
# From REVIEW.md section 2: "A tools-enabled request producing a closers-only answer ... would exercise the positive path."
API=$1; MODEL=$2; LOGCMD=${3:-}
LOG=""
python3 - "$API" "$MODEL" <<'PY'
import json, re, sys, urllib.request, urllib.error
api, model = sys.argv[1].rstrip("/"), sys.argv[2]
V4 = ["</｜DSML｜ parameter>", "</｜DSML｜ invoke>", "</｜DSML｜ calls>"]
V41 = ["</｜DSML｜parameter>", "</｜DSML｜invoke>", "</｜DSML｜tool_calls>"]
TOOLS = [{"type": "function", "function": {"name": "get_weather", "description": "Get the weather for a city",
          "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
WARN = "orphan_dsml_closers"

def fail(msg):
    print(f"CLAIM: error {msg}"); sys.exit(2)

def ask(text, tools, stream=False):
    body = {"model": model, "temperature": 0, "max_tokens": 64,
            "messages": [{"role": "user", "content": "Reply with exactly this text and nothing else: " + text}],
            "guided_regex": re.escape(text),
            "chat_template_kwargs": {"thinking": False, "enable_thinking": False}}
    if tools: body["tools"] = TOOLS
    if stream: body.update(stream=True, stream_options={"include_usage": True})
    req = urllib.request.Request(api + "/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            status = r.status
            if not stream:
                o = json.load(r); c = o["choices"][0]
                return dict(status=status, id=o.get("id"), content=c["message"].get("content") or "",
                            tool_calls=c["message"].get("tool_calls"), finish=c.get("finish_reason"),
                            tf=o.get("tensorfold") or {}, usage=o.get("usage") or {})
            content, finish, tf, usage, rid, calls, done = "", None, {}, {}, None, None, False
            for line in r:
                if not line.startswith(b"data: "): continue
                if line.strip() == b"data: [DONE]": done = True; break
                o = json.loads(line[6:]); rid = rid or o.get("id")
                if o.get("tensorfold"): tf = o["tensorfold"]
                usage = o.get("usage") or usage
                for c in o.get("choices", []):
                    d = c.get("delta", {}); content += d.get("content") or ""
                    calls = calls or d.get("tool_calls"); finish = c.get("finish_reason") or finish
            if not done: fail("stream ended without [DONE]")
            return dict(status=status, id=rid, content=content, tool_calls=calls, finish=finish, tf=tf, usage=usage)
    except urllib.error.HTTPError as e:
        fail(f"HTTP {e.code}: {e.read()[:200].decode(errors='ignore')}")
    except (urllib.error.URLError, OSError, ValueError, KeyError, IndexError) as e:
        fail(f"{type(e).__name__}: {e}"[:200])

def show(name, r):
    print(f"{name}: status={r['status']} finish={r['finish']} tool_calls={bool(r['tool_calls'])} "
          f"content={json.dumps(r['content'])} completion_tokens={r['usage'].get('completion_tokens')} "
          f"token_sha={r['tf'].get('token_sha')} tool_parse_warning={r['tf'].get('tool_parse_warning')}")

ok = True
def check(cond, what):
    global ok
    print(("  ok   " if cond else "  FAIL ") + what); ok = ok and cond

def plain(r, text):  # what the PR promises is unchanged: the content, finish reason, no invented call
    return r["status"] == 200 and r["content"] == text and r["finish"] == "stop" and not r["tool_calls"]

ids = []
for label, closers in (("V4", V4), ("V4.1", V41)):
    text = "".join(closers)
    w = ask(text, True); show(f"{label} closers, tools, whole", w)
    check(plain(w, text), f"{label} whole: 200, content is exactly the closers, finish stop, no tool_calls")
    check(w["tf"].get("tool_parse_warning") == WARN, f"{label} whole: tensorfold.tool_parse_warning == {WARN}")
    s = ask(text, True, stream=True); show(f"{label} closers, tools, stream", s)
    check(plain(s, text), f"{label} stream: content is exactly the closers, finish stop, no tool_calls")
    check(s["tf"].get("tool_parse_warning") == WARN, f"{label} stream: final event tool_parse_warning == {WARN}")
    check(s["tf"].get("token_sha") == w["tf"].get("token_sha") and
          s["usage"].get("completion_tokens") == w["usage"].get("completion_tokens"),
          f"{label}: same token_sha and completion_tokens whole vs stream")
    ids += [w["id"], s["id"]]

text = "".join(V4)
n = ask(text, False); show("V4 closers, no tools, whole", n)
check(plain(n, text) and "tool_parse_warning" not in n["tf"], "no tools: same closers, not flagged")
text = "Example: " + V4[2]
p = ask(text, True); show("prose then closer, tools, whole", p)
check(plain(p, text) and "tool_parse_warning" not in p["tf"], "prose before the closer: not flagged")

with open("/tmp/claim-0043-ids", "w") as f: f.write("\n".join(i or "" for i in ids))
sys.exit(0 if ok else 1)
PY
rc=$?
[ $rc -eq 2 ] && exit 2
if [ $rc -eq 0 ] && [ -n "$LOGCMD" ]; then
    LOG=$(bash -c "$LOGCMD" 2>/dev/null | grep 'tool_parse_warning=orphan_dsml_closers')
    echo "log lines with the warning: $(printf '%s' "$LOG" | grep -c .)"
    printf '%s\n' "$LOG" | tail -n 4 | sed 's/^/  log: /'
    while read -r id; do
        if [ -z "$id" ] || ! printf '%s' "$LOG" | grep -qF "$id"; then echo "  FAIL no log line names reply ${id:-<none>}"; rc=1; fi
    done < /tmp/claim-0043-ids
    if printf '%s' "$LOG" | grep -q 'DSML'; then echo "  FAIL the warning line carries reply text"; rc=1; fi
fi
rm -f /tmp/claim-0043-ids
if [ $rc -eq 0 ]; then echo "CLAIM: holds"; exit 0; else echo "CLAIM: does not hold"; exit 1; fi

# Run against the base build (recipe 476bbc7 + evidence commit, no 0043), 2026-11-01:
# $ ./claim.sh <API> DeepSeek-V4.1-Flash-1017
# V4 closers, tools, whole: status=200 finish=stop tool_calls=False content="</\uff5cDSML\uff5c parameter></\uff5cDSML\uff5c invoke></\uff5cDSML\uff5c calls>" completion_tokens=14 token_sha=acf0985cb22b tool_parse_warning=None
#   ok   V4 whole: 200, content is exactly the closers, finish stop, no tool_calls
#   FAIL V4 whole: tensorfold.tool_parse_warning == orphan_dsml_closers
# V4 closers, tools, stream: status=200 finish=stop tool_calls=False content="</\uff5cDSML\uff5c parameter></\uff5cDSML\uff5c invoke></\uff5cDSML\uff5c calls>" completion_tokens=14 token_sha=acf0985cb22b tool_parse_warning=None
#   ok   V4 stream: content is exactly the closers, finish stop, no tool_calls
#   FAIL V4 stream: final event tool_parse_warning == orphan_dsml_closers
#   ok   V4: same token_sha and completion_tokens whole vs stream
# V4.1 closers, tools, whole: status=200 finish=stop tool_calls=False content="</\uff5cDSML\uff5cparameter></\uff5cDSML\uff5cinvoke></\uff5cDSML\uff5ctool_calls>" completion_tokens=14 token_sha=63bef4f968dd tool_parse_warning=None
#   ok   V4.1 whole: 200, content is exactly the closers, finish stop, no tool_calls
#   FAIL V4.1 whole: tensorfold.tool_parse_warning == orphan_dsml_closers
# V4.1 closers, tools, stream: status=200 finish=stop tool_calls=False content="</\uff5cDSML\uff5cparameter></\uff5cDSML\uff5cinvoke></\uff5cDSML\uff5ctool_calls>" completion_tokens=14 token_sha=63bef4f968dd tool_parse_warning=None
#   ok   V4.1 stream: content is exactly the closers, finish stop, no tool_calls
#   FAIL V4.1 stream: final event tool_parse_warning == orphan_dsml_closers
#   ok   V4.1: same token_sha and completion_tokens whole vs stream
# V4 closers, no tools, whole: status=200 finish=stop tool_calls=False content="</\uff5cDSML\uff5c parameter></\uff5cDSML\uff5c invoke></\uff5cDSML\uff5c calls>" completion_tokens=13 token_sha=3cd6a834e61b tool_parse_warning=None
#   ok   no tools: same closers, not flagged
# prose then closer, tools, whole: status=200 finish=stop tool_calls=False content="Example: </\uff5cDSML\uff5c calls>" completion_tokens=7 token_sha=df49fda146b1 tool_parse_warning=None
#   ok   prose before the closer: not flagged
# CLAIM: does not hold
# exit=1
