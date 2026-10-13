#!/usr/bin/env python3
"""v1.3 native matched-source benchmark dispatcher, fail-closed qualification."""
from __future__ import annotations
import argparse,csv,hashlib,json,os,shutil,subprocess,sys,time,zipfile,traceback
from pathlib import Path
BASE=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(BASE/'tools'))
from core import read_json,write_json,sha256_file
VERSION='TMM_Efficiency_AutoBench_v1_3_1'
ROOT_DEFAULT='/root/autodl-tmp/TMM_Traj/source/tmm_v1'
METHODS=('proposed','tutr','eigen','vite','flow')

def go(cmd,log,timeout=900,cwd=None):
 with open(log,'a',encoding='utf8',buffering=1) as f:
  f.write('\n'+time.strftime('%Y-%m-%d %H:%M:%S')+' '+repr(cmd)+'\n');f.flush()
  try:p=subprocess.run(cmd,cwd=cwd,stdout=f,stderr=subprocess.STDOUT,timeout=timeout)
  except subprocess.TimeoutExpired:f.write('ERROR TimeoutExpired\n');return 124
  except Exception as e:f.write('ERROR '+repr(e)+'\n');return 127
  f.write('exit_code='+str(p.returncode)+'\n');return p.returncode

def hash_repr(value):return hashlib.sha256(json.dumps(value,sort_keys=True,default=str).encode()).hexdigest()

class TaskRunner:
 def __init__(self,args):
  self.a=args;self.root=Path(args.root).resolve()
  self.out=self.root/'runs'/'tmm_efficiency_autobench_v131'
  self.logs=self.out/'logs'
  self.out.mkdir(parents=True,exist_ok=True);self.logs.mkdir(parents=True,exist_ok=True)
  self.ws=self.root/'runs/tmm_efficiency_autobench_v1/workspace'
  self.scene=self.out/'canonical_univ_observations.npz'
  self.scene_meta=Path(str(self.scene)+'.manifest.json')
  self.status_path=self.out/'status.json'
  self.state=read_json(self.status_path,{'version':'1.3.1','cases':{}})
  self.state.setdefault('cases',{})
  self.base_python=self.choose_base_python()
 def choose_base_python(self):
  cands=[os.environ.get('TMM_EFF_BASE_PYTHON'),'/root/miniconda3/envs/tmm_traj/bin/python',
         '/root/autodl-tmp/tmm_env/bin/python',sys.executable]
  return next((p for p in cands if p and Path(p).is_file()),sys.executable)
 def python(self,m):
  return self.base_python if m=='proposed' else str(self.ws/'envs'/m/'bin/python')
 def ck(self,m):
  if m=='proposed':return self.root/f'runs/tmm_stage3e_ethucy_final/ethucy_5fold_social/multihyp_social/univ/seed_{self.a.seed}/best_model.pt'
  repo=self.ws/'repos'/m
  if m=='tutr':return repo/'checkpoint/univ/best.pth'
  if m=='eigen':return repo/'checkpoints/TMMeff_sgcn_univ/univ/model_best.pth'
  if m=='vite':return repo/'checkpoint/univ/univ_ckpt_best.pth'
  if m=='flow':
   marker=self.out/'adapters/flow_checkpoint_path.txt'
   if marker.is_file():
    value=marker.read_text(encoding='utf8').strip()
    return Path(value) if value else None
  return None
 def save(self):write_json(self.status_path,self.state)
 def set(self,key,state,**args):
  self.state['cases'][key]={'state':state,'time':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**args}
  self.save();print(f'[{key}] {state} {args.get("reason","")}',flush=True)
 def nums(self,total):
  out=[]
  for n in self.a.targets.split(','):
   n=n.strip().lower()
   x=total if n=='all' else int(n)
   if not 1<=x<=total:continue
   if x not in out:out.append(x)
  return out
 def preflight(self):
  r={'project_root':str(self.root),'gpu_required':'NVIDIA RTX 4090',
     'canonical_source':'official ViTE UNIV scene (same sample ID and target indices for all models)',
     'software_tested_locally':'Only source-level CPU integration and simulated paths',
     'common_fold':'univ','no_training':True,'no_weight_modification':True,
     'methods':{},'root_exists':self.root.is_dir()}
  for m in METHODS:
   ck=self.ck(m);py=self.python(m)
   r['methods'][m]={'python':py,'python_exists':Path(py).is_file(),
                     'checkpoint':str(ck),'checkpoint_exists':bool(ck and ck.is_file()),
                     'repository':str(self.ws/'repos'/m),'repository_exists':(self.ws/'repos'/m).is_dir()}
  r['methods']['flow']['checkpoint_exists']='requires original FlowChain loader inventory; not inferred from generic output folder'
  write_json(self.out/'preflight.json',r)
  return r
 def prepare(self):
  if self.a.force_scene:
   for p in (self.scene,self.scene_meta):
    if p.exists():p.unlink()
  if self.scene.exists() and self.scene_meta.exists():
   meta=read_json(self.scene_meta,{})
   if meta.get('window_index')==self.a.scene_index and int(meta.get('agents_total',0))>=self.a.min_agents:
    print('REUSE canonical scene',meta.get('observation_sha256'),flush=True);return meta
  py=self.python('vite')
  if not Path(py).exists():raise FileNotFoundError('ViTE environment missing '+py)
  cmd=[py,str(BASE/'tools/prepare_univ.py'),'--root',str(self.root),'--out',str(self.scene),
       '--index',str(self.a.scene_index),'--min-agents',str(self.a.min_agents)]
  rc=go(cmd,self.logs/'prepare_univ.log',timeout=self.a.timeout_sec,cwd=self.root)
  if rc!=0:raise RuntimeError('Failed to prepare SAME observation window, see logs/prepare_univ.log')
  meta=read_json(self.scene_meta,{})
  if not meta.get('observation_sha256'):raise RuntimeError('Scene preparation emitted no sha256')
  return meta
 def signature(self,m,n):
  ck=self.ck(m)
  if not ck or not ck.exists():return None
  return hash_repr({'model':m,'N':n,'cksha':sha256_file(ck),'scene_sha':sha256_file(self.scene),
                    'gpu':self.a.gpu,'seed':self.a.seed,'warmup':self.a.warmup,
                    'repeats':self.a.repeats,'wall_repeats':self.a.wall_repeats,
                    'adapter_sha':sha256_file(BASE/'tools/matched_adapters.py'),
                    'runner_sha':sha256_file(BASE/'tools/bench_case.py'),
                    'timing_sha':sha256_file(BASE/'tools/core.py')})
 def run_case(self,m,n):
  key=f'{m}.UNIV.N{n}';dest=self.out/f'{m}_univ_N{n}.json'
  if m=='flow' and not (self.out/'adapters/flow_full_k20.py').exists():
   self.set(key,'BLOCKED',reason='No certified same-scene K20 full sampler in official FlowChain implementation; source probe provided')
   return
  sig=self.signature(m,n)
  if not sig:
   self.set(key,'BLOCKED',reason='Original trained checkpoint missing; no random weight fallback');return
  py=self.python(m)
  if not Path(py).is_file():self.set(key,'BLOCKED',reason='Isolated method python environment absent '+str(py));return
  cached=read_json(dest,{})
  if not self.a.force and cached.get('status')=='MEASURED' and self.state['cases'].get(key,{}).get('signature')==sig:
   print('[REUSE]',key,flush=True);return
  cmd=[py,str(BASE/'tools/bench_case.py'),'--model',m,'--root',str(self.root),'--scene',str(self.scene),
       '--n',str(n),'--out',str(dest),'--seed',str(self.a.seed),'--gpu',str(self.a.gpu),
       '--warmup',str(self.a.warmup),'--repeats',str(self.a.repeats),
       '--wall-repeats',str(self.a.wall_repeats)]
  self.set(key,'RUNNING',signature=sig)
  rc=go(cmd,self.logs/(m+f'_univ_N{n}.log'),timeout=self.a.timeout_sec,cwd=self.root)
  result=read_json(dest,{})
  if rc==0 and result.get('status')=='MEASURED' and result.get('n_targets')==n and result.get('candidate_count_per_target')==20:
   self.set(key,'DONE',signature=sig,median_ms=result['gpu_forward_median_ms'],
            checkpoint_sha256=result.get('checkpoint_sha256'))
  else:self.set(key,'BLOCKED',signature=sig,reason=result.get('reason','exit code '+str(rc)),exit_code=rc)
 def flow_probe(self):
  py=sys.executable
  cmd=[py,str(BASE/'tools/flow_probe.py'),'--root',str(self.root),'--out',str(self.out)]
  rc=go(cmd,self.logs/'flow_probe.log',timeout=120,cwd=self.root)
  self.set('flow.source_probe','DONE' if rc==0 else 'FAILED',reason='full K20 native input adapter still required')
 def summarize(self):
  import collections
  meta=read_json(self.scene_meta,{})
  rows=[];blocked=[];validated=[];full=[]
  keys=['method','fold','n_targets','gpu_forward_median_ms','gpu_forward_p95_ms',
        'prepared_pipeline_wall_median_ms','prepared_pipeline_wall_p95_ms','throughput_targets_per_s',
        'parameters','candidate_count_per_target','checkpoint_sha256','physical_observation_sha256',
        'source_sha256','gpu','status','qualification']
  canonical_sha=sha256_file(self.scene) if self.scene.exists() else None
  expected_input_per_n={}
  expected_runtime_per_n={}
  for m in METHODS:
   if m=='flow' and not (self.out/'adapters/flow_full_k20.py').exists():
    blocked.append({'method':'flow','reason':'native official test K=1; K20 same-window callable not yet verified'});continue
   for n in self.nums(int(meta.get('agents_total',0))) if meta else []:
    d=read_json(self.out/f'{m}_univ_N{n}.json',{})
    if d.get('status')!='MEASURED':
     blocked.append({'method':m,'n_targets':n,'reason':d.get('reason','not run')});continue
    import numpy as np
    # identical canonical source and actual subset verified by runner for every method
    refsha=meta.get('observation_sha256')
    input_hash=d.get('physical_observation_sha256')
    if n not in expected_input_per_n:expected_input_per_n[n]=input_hash
    rt=(d.get('torch_version'),d.get('cuda_version'),d.get('gpu'))
    if n not in expected_runtime_per_n:expected_runtime_per_n[n]=rt
    runtime_matches=(rt==expected_runtime_per_n[n] and all(rt))
    source_matches=(bool(canonical_sha) and d.get('source_sha256')==canonical_sha and
                    input_hash==expected_input_per_n[n] and bool(input_hash))
    checkpoint_matches=(bool(self.ck(m)) and self.ck(m).is_file() and
                        d.get('checkpoint_sha256')==sha256_file(self.ck(m)))
    base_ok=(d.get('n_targets')==n and d.get('candidate_count_per_target')==20 and
             'RTX 4090' in d.get('gpu','') and d.get('global_coordinate_predictions_verified') is True and
             d.get('trained_checkpoint_loaded') is True and d.get('precision')=='float32' and
             d.get('tf32') is False and
             int(d.get('gpu_repeats',0))==self.a.repeats and
             int(d.get('wall_repeats',0))==self.a.wall_repeats and
             d.get('observed_steps')==8 and d.get('future_steps')==12 and
             source_matches and checkpoint_matches and runtime_matches)
    flag='SAME_INPUT_K20_VERIFIED' if base_ok else 'DIAGNOSTIC_ONLY'
    row={k:d.get(k,'') for k in keys}
    row.update(throughput_targets_per_s=1000*n/d['prepared_pipeline_wall_median_ms'],qualification=flag)
    rows.append(row)
    if base_ok: validated.append(row)
    else:blocked.append({'method':m,'n_targets':n,'reason':'qualification checks failed'})
  def csvwrite(name,items,fields):
   with open(self.out/name,'w',newline='',encoding='utf8') as fp:
    w=csv.DictWriter(fp,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(items)
  csvwrite('all_measurements.csv',rows,keys)
  csvwrite('same_scene_K20_partial.csv',validated,keys)
  csvwrite('blocked.csv',blocked,['method','n_targets','reason'])
  group=collections.defaultdict(list)
  for row in validated:group[row['n_targets']].append(row)
  noflow=[]
  for n,items in sorted(group.items()):
   models={r['method'] for r in items}
   if set(('proposed','tutr','eigen','vite')) <= models:
    base=next(r for r in items if r['method']=='proposed')
    for r in items:
     if r['method']!='proposed':
      noflow.append({'N':n,'reference':'proposed','method':r['method'],
       'gpu_forward_speedup':float(r['gpu_forward_median_ms'])/float(base['gpu_forward_median_ms']),
       'wall_pipeline_speedup':float(r['prepared_pipeline_wall_median_ms'])/float(base['prepared_pipeline_wall_median_ms'])})
  csvwrite('FOUR_METHOD_MATCHED_IF_COMPLETE.csv',noflow,['N','reference','method','gpu_forward_speedup','wall_pipeline_speedup'])
  complete_all_five=all({r['method'] for r in z}==set(METHODS) for z in group.values()) if group else False
  report=['# v1.3 Matched-Scene Native Inference Audit','',
    '**Task scope:** ONE real UNIV observation window, same physical target subset, same N and actual K=20.',
    '**Fairness:** model-specific native preprocessing retained, full model + decode. Two independent time scopes.',
    '**Important:** ViTE only released UNIV checkpoint, so 5-fold average is not possible without further trained checkpoints.',
    '**FlowChain:** old test uses nsample=1; cannot claim strict five-method result before full K20 same-window sampler adapter exists.',
    '',f"Canonical scene sha256: {meta.get('observation_sha256')}",
    f"Full source N: {meta.get('agents_total')}",
    f'Authenticated model/N results: {len(validated)}',f'Blocked model/N jobs: {len(blocked)}',
    f'**ALL FIVE STRICTLY CERTIFIED: {complete_all_five}**','',
    '## Limitations','',
    '- These are latency/throughput benchmarks only; not cross-model accuracy comparisons.',
    '- K20 and same input window are necessary but do not establish identical social-context computation: each original architecture is preserved.',
    '- Host wall timings include in-memory preprocessing and transfers, not reading files from disk.',
    '- Repeated warm model runtime is isolated from cold start and checkpoint loading.',
    '- FlowChain lacks a K20 native bridge; DO NOT use K1 or rapid density update as K20 full generator time.',
    '- A subset N changes neighborhood context for graph models; each model receives same N target set in each row.',
    '', '## Diagnostic failures']
  for q in blocked:report.append(f"- {q.get('method')} N={q.get('n_targets')}: {q.get('reason')}")
  (self.out/'REPORT.md').write_text('\n'.join(report)+'\n',encoding='utf8')
  write_json(self.out/'qualification.json',{'strict_all_five':complete_all_five,
     'same_input_k20_results':len(validated),'blocked':len(blocked),
     'four_method_matched_N':[int(z['N']) for z in noflow if z['method']=='tutr'],
     'source_N':meta.get('agents_total'),'source_hash':meta.get('observation_sha256')})
 def pack(self):
  self.summarize()
  path=self.root/'runs'/f'{VERSION}_review_bundle.zip'
  with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_DEFLATED) as z:
   for p in sorted(self.out.rglob('*')):
    if not p.is_file() or p.name in ('runner.lock',self.scene.name):continue
    if p.suffix.lower() in ('.pt','.pth','.ckpt','.npz','.pkl','.safetensors','.tar','.zip'):continue
    if p.name.endswith('.tmp'):continue
    if p.stat().st_size>3_000_000:
     if p.suffix=='.log':z.writestr(str(p.relative_to(self.out))+'__tail.txt',p.read_bytes()[-160000:])
     continue
    z.write(p,str(p.relative_to(self.out)))
  print('REVIEW_BUNDLE=',path,'bytes=',path.stat().st_size,flush=True)
 def run(self):
  if self.a.action=='status':
   print('Status',len(self.state['cases']),'cases')
   for k,v in sorted(self.state['cases'].items()):print(k,v.get('state'),v.get('reason',''))
   return
  if self.a.action=='check':
   print(json.dumps(self.preflight(),indent=2,ensure_ascii=False));return
  if self.a.action=='pack':self.pack();return
  import fcntl
  with open(self.out/'runner.lock','a+') as lock:
   fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
   print('PRECHECK',json.dumps(self.preflight(),ensure_ascii=False),flush=True)
   try:
    self.prepare()
   except Exception as exc:
    self.set('canonical_scene','FAILED',reason=str(exc))
    if 'flow' in self.a.methods:self.flow_probe()
    self.pack()
    return
   meta=read_json(self.scene_meta,{})
   for n in self.nums(int(meta['agents_total'])):
    for m in self.a.methods:
     self.run_case(m,n)
   if 'flow' in self.a.methods:self.flow_probe()
   self.pack()

def main():
 p=argparse.ArgumentParser()
 p.add_argument('action',nargs='?',choices=['run','check','pack','status'],default='run')
 p.add_argument('--root',default=os.environ.get('TMM_ROOT',ROOT_DEFAULT))
 p.add_argument('--methods',default=os.environ.get('TMM_V13_METHODS','proposed tutr eigen vite flow'))
 p.add_argument('--targets',default=os.environ.get('TMM_V13_N','1,2,4,8,16,32,all'))
 p.add_argument('--scene-index',type=int,default=int(os.environ.get('TMM_V13_SCENE_INDEX','0')))
 p.add_argument('--min-agents',type=int,default=int(os.environ.get('TMM_V13_MIN_AGENTS','32')))
 p.add_argument('--seed',type=int,default=42)
 p.add_argument('--gpu',default=os.environ.get('TMM_V13_GPU','0'));p.add_argument('--warmup',type=int,default=50)
 p.add_argument('--repeats',type=int,default=300);p.add_argument('--wall-repeats',type=int,default=60)
 p.add_argument('--timeout-sec',type=int,default=900)
 p.add_argument('--force',action='store_true');p.add_argument('--force-scene',action='store_true')
 a=p.parse_args()
 a.methods=a.methods.replace(',',' ').split()
 if any(x not in METHODS for x in a.methods):p.error('Unsupported method')
 if not Path(a.root).exists():p.error('Missing TMM project root '+a.root)
 if a.repeats<30 or a.warmup<5 or a.wall_repeats<10:p.error('Insufficient measurement rounds')
 if a.min_agents<1:p.error('min_agents <1')
 runner=TaskRunner(a)
 runner.run()
if __name__=='__main__':main()
