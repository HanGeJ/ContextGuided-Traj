#!/usr/bin/env python3
"""Independent benchmark worker in the method's *existing* virtual env."""
import os,argparse,sys,traceback,time
p=argparse.ArgumentParser()
p.add_argument('--model',required=True,choices=['proposed','tutr','eigen','vite','flow'])
p.add_argument('--root',required=True);p.add_argument('--scene',required=True)
p.add_argument('--n',type=int,required=True);p.add_argument('--out',required=True)
p.add_argument('--seed',type=int,default=42);p.add_argument('--gpu',default='0')
p.add_argument('--warmup',type=int,default=50);p.add_argument('--repeats',type=int,default=300)
p.add_argument('--wall-repeats',type=int,default=60)
a=p.parse_args()
os.environ['CUDA_VISIBLE_DEVICES']=a.gpu
os.environ['PYTHONHASHSEED']='42'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
sys.path.insert(0,os.path.dirname(__file__))
from core import write_json,sha256_bytes,sha256_file,assert_k20,timing
from pathlib import Path
import numpy as np
try:
 obs=np.load(a.scene,allow_pickle=False)['obs']
 if obs.dtype!=np.float32 or obs.ndim!=3 or obs.shape[1:]!=(8,2):
  raise ValueError('Canonical data must be float32 [N,8,2]')
 if not(1<=a.n<=len(obs)):raise ValueError(f'Invalid N {a.n}, source N={len(obs)}')
 # Stable, geometry-independent deterministic target choice; same for all methods.
 inds=np.arange(len(obs)) if a.n==len(obs) else np.linspace(0,len(obs)-1,a.n,dtype=np.int64)
 if len(np.unique(inds))!=a.n:raise ValueError('Repeated target indexes')
 sample=np.ascontiguousarray(obs[inds])
 obs_sha=sha256_bytes(sample.tobytes())
 import torch
 if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
 torch.set_grad_enabled(False)
 torch.backends.cuda.matmul.allow_tf32=False
 torch.backends.cudnn.allow_tf32=False
 import matched_adapters
 adapter=getattr(matched_adapters,a.model)(a.root,'univ',a.seed)
 model=adapter['model']
 if model.training:raise ValueError('Model must be eval()')
 prepared=adapter['prepare'](sample)
 test=adapter['forward'](prepared)
 assert_k20(test,a.n,adapter['expected'])
 if a.model=='flow':
  # Reject copied one-sample predictions masquerading as K20. The adapter
  # must also identify and use the trained official sampling function.
  canon=(test if adapter['expected']=='NKT2' else
         test.transpose(0,1) if adapter['expected']=='KNT2' else test[0])
  delta=(canon[:,1:,:,:]-canon[:,:1,:,:]).abs().amax().item()
  if delta<1e-6:raise RuntimeError('FlowChain output duplicates K1 across all 20 modes')
 del prepared,test
 metrics=timing(torch,lambda: adapter['prepare'](sample),adapter['forward'],
                lambda y: assert_k20(y,a.n,adapter['expected']),
                warmup=a.warmup,repeats=a.repeats,wall_repeats=a.wall_repeats)
 sample_gpu=metrics.pop('gpu_samples_ms');sample_wall=metrics.pop('wall_samples_ms')
 from core import write_json
 meta={'status':'MEASURED','method':a.model,'fold':'univ','data_source':'ViTE official UNIV window, observation-only',
       'source_manifest':str(Path(a.scene).resolve())+'.manifest.json',
       'source_sha256':sha256_file(a.scene),
       'physical_observation_sha256':obs_sha,
       'n_targets':a.n,'target_indices':inds.tolist(),'candidate_count_per_target':20,
       'observed_steps':8,'future_steps':12,'output_layout':adapter['expected'],
       'input_geometry':'same observed absolute trajectories in ETH-UCY scene coordinates; per-method native transformations',
       'checkpoint_path':str(adapter['checkpoint']),
       'checkpoint_sha256':sha256_file(adapter['checkpoint']),
       'parameters':sum(v.numel() for v in model.parameters()),
       'notes':adapter['notes'], 'trained_checkpoint_loaded':True,
       'global_coordinate_predictions_verified':True,
       'n_agents_matches_canonical':True,**metrics}
 write_json(a.out,meta)
 write_json(str(a.out)+'.samples.json',{'model_only_gpu_ms':sample_gpu,'prepared_pipeline_wall_ms':sample_wall,
                                       'data_hash':obs_sha,'n_targets':a.n})
 print('SUCCESS',a.model,'N',a.n,'GPU median',meta['gpu_forward_median_ms'],
       'pipeline median',meta['prepared_pipeline_wall_median_ms'],flush=True)
except Exception as e:
 err={'status':'BLOCKED','model':a.model,'fold':'univ','n_targets':a.n,
      'reason':f'{type(e).__name__}: {e}','traceback':traceback.format_exc()[-12000:],
      'message':'No latency may be included in a strict comparison for this case.'}
 write_json(a.out,err)
 print('BLOCKED',a.model,a.n,err['reason'],flush=True)
 sys.exit(3)
