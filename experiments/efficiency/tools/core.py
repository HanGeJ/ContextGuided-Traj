"""Safe, self-contained helpers for matched-scene forward benchmarking."""
from __future__ import annotations
import hashlib,json,os,time,math,statistics
from pathlib import Path

def sha256_bytes(data): return hashlib.sha256(data).hexdigest()
def sha256_file(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def write_json(path,obj):
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
 tmp=p.with_name(p.name+'.tmp')
 with open(tmp,'w',encoding='utf8') as f:json.dump(obj,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
 os.replace(tmp,p)
def read_json(path,default=None):
 try:return json.loads(Path(path).read_text(encoding='utf8'))
 except (OSError,ValueError,TypeError):return default

def q95(v):
 vals=sorted(v); i=.95*(len(vals)-1); lo=int(i); hi=min(lo+1,len(vals)-1)
 return float(vals[lo]+(i-lo)*(vals[hi]-vals[lo]))

def timing(torch,prepare,forward,verify,warmup=50,repeats=300,wall_repeats=60):
 """Two *different* scopes: model-only GPU-resident input; host+GPU preparation/inference wallclock.

 No file IO or checkpoint load in either scope; no synthetic model outputs.
 """
 if not torch.cuda.is_available():raise RuntimeError('RTX GPU/CUDA required')
 torch.backends.cuda.matmul.allow_tf32=False
 torch.backends.cudnn.allow_tf32=False
 assert repeats>=30 and warmup>=5 and wall_repeats>=10
 with torch.inference_mode():
  prepared=prepare()
  for _ in range(warmup): output=forward(prepared)
  torch.cuda.synchronize()
  verify(output)
  torch.cuda.reset_peak_memory_stats()
  gpu=[]
  starts=[torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
  ends=[torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
  for i in range(repeats):
   starts[i].record();output=forward(prepared);ends[i].record()
   torch.cuda.synchronize()
   gpu.append(starts[i].elapsed_time(ends[i]))
  verify(output)
  peak_cuda_model=torch.cuda.max_memory_allocated()/1048576
  for _ in range(min(15,warmup)):
   x=prepare();output=forward(x)
  torch.cuda.synchronize()
  torch.cuda.reset_peak_memory_stats()
  wall=[]
  for _ in range(wall_repeats):
   torch.cuda.synchronize();t=time.perf_counter()
   x=prepare();output=forward(x)
   torch.cuda.synchronize();wall.append((time.perf_counter()-t)*1000.)
  verify(output)
  peak_cuda_full=torch.cuda.max_memory_allocated()/1048576
 return {
  'gpu_forward_median_ms':float(statistics.median(gpu)),
  'gpu_forward_p95_ms':q95(gpu),
  'gpu_forward_mean_ms':float(statistics.mean(gpu)),
  'prepared_pipeline_wall_median_ms':float(statistics.median(wall)),
  'prepared_pipeline_wall_p95_ms':q95(wall),
  'prepared_pipeline_wall_mean_ms':float(statistics.mean(wall)),
  'peak_cuda_model_MiB':peak_cuda_model,
  'peak_cuda_full_MiB':peak_cuda_full,
  'gpu_samples_ms':gpu,
  'wall_samples_ms':wall,
  'gpu_repeats':len(gpu),'wall_repeats':len(wall),'warmup':warmup,
  'timing_scope_gpu':'GPU-resident native full network with output decoding',
  'timing_scope_wall':'in-memory shared observation -> per-model preprocessing and device transfer -> full model -> output decoding; excludes disk IO and checkpoint init',
  'gpu':str(torch.cuda.get_device_name(0)),'cuda_version':str(torch.version.cuda),
  'torch_version':str(torch.__version__),
  'precision':'float32','tf32':False,
  }

def assert_k20(tensor,n,layout):
 import torch
 if not isinstance(tensor,torch.Tensor) or not tensor.is_cuda:raise RuntimeError('Inference did not produce CUDA tensor')
 if layout=='NKT2': expected=(n,20,12,2)
 elif layout=='KNT2': expected=(20,n,12,2)
 elif layout=='BNKT2':expected=(1,n,20,12,2)
 else: raise ValueError(layout)
 if tuple(tensor.shape)!=expected:raise RuntimeError(f'Wrong output shape: {tuple(tensor.shape)} expected {expected}')
 if not bool(torch.isfinite(tensor).all()):raise RuntimeError('Nonfinite predictions')
