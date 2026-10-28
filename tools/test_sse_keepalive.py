"""Exercise the native HTTP handlers against the existing scripted CPU engine."""
import argparse
import http.client
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time

parser = argparse.ArgumentParser()
parser.add_argument('variant', choices=['baseline', 'candidate'])
parser.add_argument('--binary', type=Path, required=True)
parser.add_argument('--fixtures', type=Path, required=True)
args = parser.parse_args()
binary = args.binary.resolve()
fixtures = args.fixtures.resolve()
results = []

def request(port, path, body, leave=False):
    conn = http.client.HTTPConnection('127.0.0.1', port, timeout=10)
    started = time.monotonic()
    conn.request('POST', path, json.dumps(body), {'Content-Type': 'application/json'})
    response = conn.getresponse()
    frames = []
    if not body.get('stream') or response.status != 200:
        data = response.read().decode()
    else:
        lines = []
        while line := response.fp.readline():
            lines.append(line.decode())
            frames.append((time.monotonic() - started, line.decode().strip()))
            if leave and ('keepalive' in lines[-1] or 'event: ping' in lines[-1]):
                break
        data = ''.join(lines)
    response.close()
    conn.close()
    return response.status, data, frames

with tempfile.TemporaryDirectory(prefix='tf-heartbeat-') as temp:
    temp = Path(temp)
    scripts = json.loads((fixtures/'scripts.json').read_text())
    scripts['heartbeat'] = {'delay_ms': 600, 'chunks': [['Hello'], [' world'], ['<|im_end|>']]}
    (temp/'scripts.json').write_text(json.dumps(scripts))
    for interval in ['0.1', '0']:
        env = os.environ | {'TF_SSE_KEEPALIVE_S': interval, 'TF_FAKE_SCRIPTS': str(temp/'scripts.json'), 'TENSORFOLD_NO_LIVE': '1'}
        with (temp/'server.log').open('w+') as log:
            proc = subprocess.Popen([str(binary), 'serve', str(fixtures), '--host', '127.0.0.1', '--port', '0', '--name', 'test-model', '--no-thinking'], env=env, stdout=log, stderr=log)
            try:
                deadline = time.monotonic()+10
                port = None
                while time.monotonic() < deadline:
                    log.seek(0)
                    if match := re.search(r'PORT (\d+)', log.read()):
                        port = int(match.group(1))
                        break
                    assert proc.poll() is None, 'fixture exited before listening'
                    time.sleep(0.05)
                assert port, 'fixture did not listen'
                routes = ['chat', 'messages', 'responses'] if interval != '0' else ['chat']
                for route in routes:
                    body = {'model':'test-model', 'max_tokens':50, 'stream':True, 'messages':[{'role':'user','content':'@script=heartbeat'}]}
                    path = '/v1/chat/completions'
                    if route == 'messages': path = '/v1/messages'
                    if route == 'responses':
                        path = '/v1/responses'
                        body['input'] = body.pop('messages')
                    status, raw, frames = request(port, path, body)
                    assert status == 200, (route, status, raw)
                    first = raw.index('Hello')
                    beats = raw[:first].count('event: ping') if route == 'messages' else raw[:first].count(': keepalive')
                    expected = args.variant == 'candidate' and interval != '0'
                    assert (beats >= 2) if expected else beats == 0, (route, interval, beats)
                    results.append({'route':route,'interval':interval,'pre_token_heartbeats':beats,'seconds':round(frames[-1][0],3)})
                if interval != '0':
                    body = {'model':'test-model','max_tokens':50,'stream':True,'messages':[{'role':'user','content':'@script=heartbeat'}], 'tools':[{'type':'function','function':{'name':'lookup','parameters':{'type':'object'}}}]}
                    status, raw, frames = request(port,'/v1/chat/completions',body)
                    beats = raw[:raw.index('Hello')].count(': keepalive')
                    assert status == 200 and (beats >= 2 if args.variant == 'candidate' else beats == 0)
                    results.append({'route':'chat-tools','pre_token_heartbeats':beats})
                    status, raw, _ = request(port,'/v1/chat/completions',{'model':'test-model','stream':True})
                    assert status == 400 and ': keepalive' not in raw
                    results.append({'route':'invalid-preflight','status':status})
                    body.pop('tools')
                    body['stream'] = False
                    status, raw, _ = request(port,'/v1/chat/completions',body)
                    assert status == 200 and ': keepalive' not in raw and json.loads(raw)['choices'][0]['message']['content'] == 'Hello world'
                    results.append({'route':'nonstream','status':status})
                    if args.variant == 'candidate':
                        body['stream'] = True
                        request(port,'/v1/chat/completions',body,leave=True)
                        time.sleep(0.7)
                        body['messages'][0]['content'] = '@script=plain'
                        status, raw, _ = request(port,'/v1/chat/completions',body)
                        assert status == 200 and 'Hello world' in raw
                        log.flush()
                        log.seek(0)
                        assert 'reason=client-left' in log.read(), 'disconnect did not cancel the scripted request'
                        results.append({'route':'disconnect-then-next','status':status})
            finally:
                proc.terminate()
                proc.wait(timeout=5)
print(json.dumps({'variant':args.variant,'tests':results},indent=2))
