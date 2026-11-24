"""Run with DATA (ordered-prompt checkout), MODELS, SERVER and RUN (output directory)."""
import concurrent.futures as cf
import hashlib, itertools, json, os, socket, subprocess, time
from pathlib import Path
from urllib.request import Request, urlopen

def hashtext(text):
    return hashlib.sha256(text.encode()).hexdigest()

def normalized(text):
    return ' '.join(text.casefold().split())

def roster(source,task):
    def rows(split):
        seen=set()
        for line in (source/'data'/task/(split+'.jsonl')).read_text().splitlines():
            r=json.loads(line); s=r['sentence'].strip(); k=normalized(s)
            if s and len(s)<=220 and k not in seen:
                seen.add(k)
                yield s,int(r['label'])+(task=='sst2')
    train=list(rows('train')); test=list(rows('dev_subsample'))
    overlap={normalized(s) for s,_ in train}&{normalized(s) for s,_ in test}
    def rank(rows,salt):
        return sorted((r for r in rows if normalized(r[0]) not in overlap),key=lambda r:hashtext('demo-v1|'+task+salt+r[0]))
    train=rank(train,'train');test=rank(test,'test')[:64];assert len(test)==64
    labels=(1,2,1,2) if task=='sst2' else (1,2,3,4)
    pools={i:iter([r for r in train if r[1]==i]) for i in labels}
    groups = [([next(pools[i]) for i in labels],test[j:j+8]) for j in range(0,64,8)]
    assert len({normalized(s) for d,_ in groups for s,_ in d})==32
    assert all(sorted(y for _,y in d)==sorted(labels) for d,_ in groups)
    return groups

def predict(port,prompt,classes):
    data={'prompt':prompt,'grammar':'root ::= ['+classes+']','temperature':0,'n_predict':1,'cache_prompt':True,'seed':42}
    with urlopen(Request(f'http://127.0.0.1:{port}/completion',data=json.dumps(data).encode(),headers={'Content-Type':'application/json'}),timeout=120) as f:
        r=json.load(f)
    assert not r.get('truncated') and r['content'] in tuple(classes)
    return int(r['content'])

def main():
    source,models,server,out=[Path(os.environ[k]) for k in ('DATA','MODELS','SERVER','RUN')]
    assert os.environ.get('RIG_GPU_RUN_ID') and os.environ['CUDA_VISIBLE_DEVICES']==os.environ['RIG_EXPECTED_GPU_UUID']
    allrows={}
    for name,filename in [('qwen','qwen2.5-3b-instruct-q4_k_m.gguf'),('smol','SmolLM3-Q4_K_M.gguf')]:
        with socket.socket() as sock:
            sock.bind(('127.0.0.1',0)); port=sock.getsockname()[1]
        with (out/(name+'.log')).open('a') as log:
            p=subprocess.Popen([server,'-m',models/filename,'-ngl','99','-c','4096','-np','8','--port',str(port),'--host','127.0.0.1'],stdout=log,stderr=log)
            try:
                for _ in range(120):
                    assert p.poll() is None
                    try:
                        with urlopen(f'http://127.0.0.1:{port}/health',timeout=1): break
                    except OSError: time.sleep(1)
                live=subprocess.check_output(['nvidia-smi','-i',os.environ['CUDA_VISIBLE_DEVICES'],'--query-compute-apps=pid','--format=csv,noheader'],text=True)
                assert str(p.pid) in live.split()
                for task,header,classes in [('sst2','Classify sentiment. 1 = negative, 2 = positive.','12'),('agnews','Classify the news. 1 = world, 2 = sports, 3 = business, 4 = science/technology.','1234')]:
                    for g,(demos,queries) in enumerate(roster(source,task)):
                        key=f'{name}-{task}-{g}'; file=out/(key+'.json')
                        if not file.exists():
                            records=[]
                            with cf.ThreadPoolExecutor(max_workers=8) as pool:
                                for perm in itertools.permutations(demos):
                                    prefix=header+'\n'+''.join(f'Text: {s}\nLabel: {y}\n' for s,y in perm)
                                    prompts=[prefix+'Text: '+s+'\nLabel:' for s,_ in queries]
                                    records.append(list(pool.map(lambda prompt:predict(port,prompt,classes),prompts)))
                            file.write_text(json.dumps(records))
                        allrows[key]=json.loads(file.read_text())
                        assert len(allrows[key])==24 and all(len(row)==8 for row in allrows[key])
            finally:
                p.terminate();p.wait(timeout=30)
    (out/'predictions.json').write_text(json.dumps(allrows))
if __name__=='__main__': main()
