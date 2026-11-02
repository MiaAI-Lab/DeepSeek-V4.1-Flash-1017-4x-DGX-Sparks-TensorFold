#!/usr/bin/env bash
# Written by Hector Isaza (hecisaza, https://github.com/hecisaza) with the verify step of inference-agent-kit
# (https://github.com/hecisaza/inference-agent-kit). Apache-2.0. Unchanged below this header from
# https://github.com/hecisaza/DeepSeek-V4.1-Flash-1017-4x-DGX-Sparks-TensorFold/blob/f698c2fe71b6a9920a5ff7646302265b524231e1/claims/MiaAI-Lab-DeepSeek-V4.1-Flash-1017-4x-DGX-Sparks-TensorFold-pr-31.sh
# Usage: tools/claims/0042-json-answer-channel.sh API MODEL. Exit 0: the claim holds. 1: it does not. 2: the script could not judge.
# claim: patches: 0042 a JSON answer stays JSON — a response_format json_schema / guided_json reply that parses as a
#   bare tool call ({"name": ..., "arguments": ...}) is returned as the message content with finish_reason "stop",
#   not converted to tool_calls, when the same request also offers `tools`; whole and streamed. Ordinary tool calls
#   (same tools, no response grammar) are still tool calls.
# How: one tool (get_weather) offered; a strict json_schema that forces exactly {"name":"get_weather","arguments":
#   {"city":"Oslo"}}, sent whole and streamed, and the same schema as guided_json (whole). Each must be HTTP 200 with
#   that JSON as content, no tool_calls, finish_reason "stop". Control: the same tools without a grammar and a
#   weather question must still come back as a get_weather tool call (finish_reason "tool_calls").
# From REVIEW.md §2: the decisive condition is "structured JSON plus offered tools, especially JSON resembling
#   {"name":…,"arguments":…}", and structured JSON streaming; neither ran in the suites.
API=$1; MODEL=$2
python3 - "$API" "$MODEL" <<'PY'
import json, sys, urllib.request, urllib.error
api, model = sys.argv[1].rstrip("/"), sys.argv[2]
TOOLS = [{"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city",
          "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
SCHEMA = {"type": "object", "properties": {
            "name": {"type": "string", "enum": ["get_weather"]},
            "arguments": {"type": "object", "properties": {"city": {"type": "string", "enum": ["Oslo"]}},
                          "required": ["city"], "additionalProperties": False}},
          "required": ["name", "arguments"], "additionalProperties": False}
EXPECT = {"name": "get_weather", "arguments": {"city": "Oslo"}}
ASK = "Describe, as JSON with a name and arguments, the call that would get the weather in Oslo."

def err(why):
    print(f"CLAIM: error {why}"); sys.exit(2)

def send(extra, messages, stream):
    body = {"model": model, "temperature": 0, "max_tokens": 64, "tools": TOOLS, "messages": messages,
            "chat_template_kwargs": {"thinking": False, "enable_thinking": False}}
    body.update(extra)
    if stream: body["stream"] = True
    req = urllib.request.Request(api + "/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            status = r.status
            if not stream:
                o = json.load(r); ch = (o.get("choices") or [{}])[0]; m = ch.get("message") or {}
                return status, m.get("content"), m.get("tool_calls") or [], ch.get("finish_reason")
            content, calls, finish, done = "", [], None, False
            for line in r:
                if not line.startswith(b"data: "): continue
                if line.strip() == b"data: [DONE]": done = True; break
                for c in json.loads(line[6:]).get("choices", []):
                    d = c.get("delta") or {}
                    content += d.get("content") or ""
                    calls += d.get("tool_calls") or []
                    finish = c.get("finish_reason") or finish
            if not done: err("stream ended without [DONE]")
            return status, content, calls, finish
    except urllib.error.HTTPError as e:
        return e.code, e.read()[:200].decode(errors="ignore"), [], None
    except (urllib.error.URLError, OSError, ValueError) as e:
        err(f"{type(e).__name__}")

def is_expected_json(s):
    try: return json.loads(s) == EXPECT
    except (TypeError, ValueError): return False

ok = True
rf = {"response_format": {"type": "json_schema", "json_schema": {"name": "planned_call", "strict": True, "schema": SCHEMA}}}
cases = [("json_schema whole", rf, False), ("json_schema stream", rf, True), ("guided_json whole", {"guided_json": SCHEMA}, False)]
for label, extra, stream in cases:
    status, content, calls, finish = send(extra, [{"role": "user", "content": ASK}], stream)
    good = status == 200 and is_expected_json(content) and not calls and finish == "stop"
    print(f"{label}: status={status} finish={finish} tool_calls={len(calls)} content={json.dumps(content)[:160]} -> {'ok' if good else 'FAIL'}")
    ok &= good

# control: ordinary tool call beside the same tools is unchanged
status, content, calls, finish = send({}, [{"role": "user", "content": "What is the weather in Oslo right now?"}], False)
names = [(c.get("function") or {}).get("name") for c in calls]
good = status == 200 and names == ["get_weather"] and finish == "tool_calls"
print(f"control tool call: status={status} finish={finish} calls={names} -> {'ok' if good else 'FAIL'}")
if status != 200: err(f"control request returned HTTP {status}")
ok &= good

print("CLAIM: holds" if ok else "CLAIM: does not hold")
sys.exit(0 if ok else 1)
PY

# Output against the base build (recipe 476bbc7, without 0042):
# json_schema whole: status=200 finish=tool_calls tool_calls=1 content=null -> FAIL
# json_schema stream: status=200 finish=tool_calls tool_calls=2 content="{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Oslo\"}}" -> FAIL
# guided_json whole: status=200 finish=tool_calls tool_calls=1 content=null -> FAIL
# control tool call: status=200 finish=tool_calls calls=['get_weather'] -> ok
# CLAIM: does not hold
# (exit 1)
