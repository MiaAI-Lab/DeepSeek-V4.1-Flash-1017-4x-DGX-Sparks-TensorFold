"""Check cached-image replay and late admission before/after a native-server restart; no image data is saved."""
import argparse
import concurrent.futures
import json
import base64
import mimetypes
from pathlib import Path
import threading
import time
import urllib.request

p = argparse.ArgumentParser()
p.add_argument('--image', action='append', required=True, help='Original image file; repeat for multiple sources')
p.add_argument('--url',default='http://127.0.0.1:8889/v1')
p.add_argument('--model',default='DeepSeek-V4.1-Flash-1017')
p.add_argument('--output', required=True)
p.add_argument('--compare')
p.add_argument('--rows',type=int,default=2000)
p.add_argument('--image-count',type=int,default=35)
args = p.parse_args()
images = [{'media_type':mimetypes.guess_type(path)[0] or 'image/png','data':base64.b64encode(Path(path).read_bytes()).decode()} for path in args.image]
if args.image_count:
    images = (images*((args.image_count+len(images)-1)//len(images)))[:args.image_count]
base = {'model':args.model,'temperature':0,'seed':42,'return_token_ids':True,
        'chat_template_kwargs':{'enable_thinking':False},'stream':True,'stream_options':{'include_usage':True}}
result = {'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'tests':{}}

def stream(body, first=None):
    start = time.monotonic()
    stamps, beats = [], []
    runtime, usage = {}, {}
    req = urllib.request.Request(args.url.rstrip('/')+'/chat/completions',json.dumps(body).encode(),{'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=600) as response:
        for raw in response:
            line = raw.decode().strip()
            if line.startswith(':'): beats.append(time.monotonic()-start)
            if not line.startswith('data:') or line == 'data: [DONE]': continue
            d = json.loads(line[5:])
            assert 'error' not in d,d
            runtime = d.get('tensorfold') or runtime
            usage = d.get('usage') or usage
            for choice in d.get('choices',[]):
                delta = choice.get('delta') or {}
                if chunk := delta.get('content') or delta.get('reasoning_content'):
                    stamps.append(time.monotonic())
                    if first is not None: first.set()
    return {'start':start,'seconds':time.monotonic()-start,'first_seconds':stamps[0]-start,
            'event_times':stamps,'heartbeats':beats,'runtime':runtime,'usage':usage,
            'max_gap_seconds':max((b-a for a,b in zip(stamps,stamps[1:])),default=0)}

def save(name, row):
    result['tests'][name] = row
    path = Path(args.output)
    path.write_text(json.dumps(result,indent=2));path.chmod(0o600)
    print(json.dumps({'name':name,'seconds':row['seconds'],'first_seconds':row['first_seconds'],
                      'max_gap_seconds':row['max_gap_seconds'],'usage':row['usage'],
                      'prefill_seconds':row['runtime']['prefill_seconds'],'token_sha':row['runtime']['token_sha']}),flush=True)

content = [{'type':'image_url','image_url':{'url':'data:'+x['media_type']+';base64,'+x['data']}} for x in images]
content.append({'type':'text','text':'Retained-image latency verification.\n'+
                '\n'.join(f'Item {i}: fox {i} jumps over dog {i*7}.' for i in range(args.rows))+
                '\nDescribe the images, including layout and colors, in sixty words.'})
picture = base | {'messages':[{'role':'user','content':content}],'max_tokens':128}
for name in ('image-cold','image-replay'):
    save(name,stream(picture))
assert result['tests']['image-cold']['runtime']['token_ids']==result['tests']['image-replay']['runtime']['token_ids']
assert result['tests']['image-replay']['usage']['prompt_tokens_details']['cached_tokens']>0.9*result['tests']['image-replay']['usage']['prompt_tokens']

decoder = base | {'messages':[{'role':'user','content':'List the integers from 1 to 1000, separated by commas. Do not skip any.'}],
                  'max_tokens':768,'ignore_eos':True}
first = threading.Event()
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
    active = pool.submit(stream,decoder,first)
    assert first.wait(300),'decoder did not start'
    replay = pool.submit(stream,picture)
    save('overlap-image',replay.result())
    save('overlap-decoder',active.result())
assert result['tests']['overlap-image']['runtime']['token_ids']==result['tests']['image-cold']['runtime']['token_ids']

long = base | {'messages':[{'role':'user','content':'Late-arrival latency verification.\n'+
    '\n'.join(f'Record {i}: cobalt goes beside amber {i*13}.' for i in range(4000))+'\nReply only with OK.'}],'max_tokens':8}
short = base | {'messages':[{'role':'user','content':'Reply only with OK.'}],'max_tokens':8}
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
    ongoing = pool.submit(stream,long)
    time.sleep(1)
    arriving = pool.submit(stream,short)
    save('late-short',arriving.result())
    save('late-long',ongoing.result())
if args.compare:
    before = json.loads(Path(args.compare).read_text())['tests']
    for name,row in result['tests'].items():
        assert row['runtime']['token_ids']==before[name]['runtime']['token_ids'],('changed tokens',name)
    short, long = result['tests']['late-short'],result['tests']['late-long']
    assert short['start']+short['first_seconds']<long['start']+long['first_seconds'],'short request still waited for long prefill'
    assert result['tests']['image-replay']['runtime']['prefill_seconds'] < before['image-replay']['runtime']['prefill_seconds']*.5,'cached image latency did not improve'
result['passed'] = True
Path(args.output).write_text(json.dumps(result,indent=2))
print('PASS: exactness and applicable admission check',flush=True)
