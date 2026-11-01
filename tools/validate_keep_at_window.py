"""Check TF_DS_KEEP_AT_WINDOW: a prompt's cold and warm replies must be the same tokens, its warm pass shorter.

Each prompt is sent twice in a row (cold, then immediately again: warm). The replies are compared by the server's
`tensorfold.token_sha` and by their token ids, and the prompt tokens each pass restored from a kept state are read from
`usage.prompt_tokens_details.cached_tokens`. A kept state at the start of the last sliding window may not change what
the model answers: a warm reply that differs from its cold one is a failure, and so is a warm pass that restored no
more than its cold one (the state was refused, or dropped by a later gate).

    python3 tools/validate_keep_at_window.py --url http://127.0.0.1:8889/v1
    python3 tools/validate_keep_at_window.py --image a.png --image-count 20 --output /tmp/keep-at-window.json

The default text prompts sit well below and above the prompt cache's 4096-token floor; the image prompt re-encodes its
last picture (it lies inside the sliding window), so its gain is the pictures before that one.
"""
import argparse
import base64
import json
import mimetypes
import time
import urllib.request
from pathlib import Path

FILLER = ('The river ran past the mill while the quick brown fox crossed the yard, and the dog that had been sleeping '
          'under the cart woke, stretched, and followed it to the gate. ')

p = argparse.ArgumentParser()
p.add_argument('--url', default='http://127.0.0.1:8889/v1')
p.add_argument('--model', default='DeepSeek-V4.1-Flash-1017')
p.add_argument('--image', action='append', default=[], help='Picture file; repeat for several')
p.add_argument('--image-count', type=int, default=0, help='Pictures in the image prompt (0: no image prompt)')
p.add_argument('--text-tokens', type=int, action='append', default=[], help='Text-only prompt sizes, default 900, 3000, 9500')
p.add_argument('--max-tokens', type=int, default=64)
p.add_argument('--output', help='Write the measurements as JSON here')
args = p.parse_args()
text_tokens = args.text_tokens or [900, 3000, 9500]


def text(n):
    """Roughly n tokens of prose (about 4 characters a token)."""
    return (FILLER * (4 * n // len(FILLER) + 1))[:4 * n]


def pictures(n):
    data = [{'type': 'image_url', 'image_url': {'url': 'data:%s;base64,%s' % (
        mimetypes.guess_type(path)[0] or 'image/png', base64.b64encode(Path(path).read_bytes()).decode())}}
        for path in args.image]
    return (data * ((n + len(data) - 1) // len(data)))[:n]


def run(messages):
    """One streaming request: its reply's token_sha, token ids, cached prompt tokens and first-token time."""
    body = {'model': args.model, 'messages': messages, 'max_tokens': args.max_tokens, 'temperature': 0, 'seed': 42,
            'return_token_ids': True, 'chat_template_kwargs': {'enable_thinking': False},
            'stream': True, 'stream_options': {'include_usage': True}}
    req = urllib.request.Request(args.url.rstrip('/') + '/chat/completions', json.dumps(body).encode(),
                                 {'Content-Type': 'application/json'})
    started = time.monotonic()
    first, runtime, usage, ids, text_ = None, {}, {}, [], []
    with urllib.request.urlopen(req, timeout=1800) as response:
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith('data:') or line == 'data: [DONE]':
                continue
            d = json.loads(line[5:])
            if 'error' in d:
                raise SystemExit(d['error'])
            runtime = d.get('tensorfold') or runtime
            usage = d.get('usage') or usage
            for choice in d.get('choices', []):
                delta = choice.get('delta') or {}
                if delta.get('content') or delta.get('reasoning_content'):
                    first = first or time.monotonic()
                    text_.append(delta.get('content') or delta.get('reasoning_content'))
                ids.extend(delta.get('token_ids') or [])
    return {'token_sha': runtime.get('token_sha'), 'tokens': len(ids), 'first_seconds': None if first is None else first - started,
            'seconds': time.monotonic() - started, 'prompt_tokens': usage.get('prompt_tokens'),
            'cached': (usage.get('prompt_tokens_details') or {}).get('cached_tokens'), 'text': ''.join(text_)}


cases = [('text %d tokens' % n, [{'role': 'user', 'content': text(n) + '\nReply with one sentence about it.'}])
         for n in text_tokens]
if args.image_count:
    if not args.image:
        raise SystemExit('--image-count needs at least one --image')
    cases.append(('images %d' % args.image_count,
                  [{'role': 'user', 'content': pictures(args.image_count) + [
                      {'type': 'text', 'text': 'Reply with one sentence about these pictures.'}]}]))

result, failed = {'cases': {}}, []
for name, messages in cases:
    cold = run(messages)
    warm = run(messages)
    same = cold['token_sha'] == warm['token_sha']
    kept = warm['cached'] is not None and cold['cached'] is not None and warm['cached'] > cold['cached']
    result['cases'][name] = {'cold': cold, 'warm': warm, 'same_reply': same, 'warm_restored_more': kept}
    print('%-18s prompt %5s  cold %6s cached, %.2f s  warm %6s cached, %.2f s  %s  %s' % (
        name, cold['prompt_tokens'], cold['cached'], cold['seconds'], warm['cached'], warm['seconds'],
        'same' if same else 'DIFFERENT (%s vs %s)' % (cold['token_sha'], warm['token_sha']),
        '' if kept else '(warm restored no more than cold)'), flush=True)
    if not same:
        failed.append(name + ': reply')
    if not kept:
        failed.append(name + ': cached')
if args.output:
    path = Path(args.output)
    path.write_text(json.dumps(result, indent=2))
    path.chmod(0o600)
print('%d of %d prompts passed' % (len(cases) - len(failed), len(cases)), flush=True)
raise SystemExit(1 if failed else 0)
