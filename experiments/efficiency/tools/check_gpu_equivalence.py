#!/usr/bin/env python3
"""Additional whole-model sanity check against the original v1.3 neighbor implementation.

Uses an existing real canonical observation window. Loads original Stage 3E weights,
compares old/reference and vectorized neighbor encodings and *full model outputs*.
No training, data writing or checkpoint updates.
"""
from __future__ import annotations
import argparse,sys,json,hashlib
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--scene',required=True);p.add_argument('--out',required=True)
a=p.parse_args()
sys.path.insert(0,str(Path(__file__).resolve().parent))
from matched_adapters import proposed
from neighbor_fast import neighbor_rel_reference,neighbor_rel_vectorized
import torch
if not torch.cuda.is_available():raise RuntimeError('GPU equivalence test requires CUDA; no result certified')
torch.set_grad_enabled(False)
obs=np.load(a.scene,allow_pickle=False)['obs']
ad=proposed(a.root,'univ',42)
train=Path(a.root)/'data/ethucy_5fold_social/univ/train.npz'
with np.load(train,allow_pickle=False) as f:topk=int(f['neighbor_rel'].shape[2])
check=[]
for n in (1,4,16,len(obs)):
 if n>len(obs):continue
 indexes=np.arange(len(obs)) if n==len(obs) else np.linspace(0,len(obs)-1,n,dtype=np.int64)
 x=np.ascontiguousarray(obs[indexes].copy())
 r0,m0=neighbor_rel_reference(x,topk)
 r1,m1=neighbor_rel_vectorized(x,topk)
 if not (np.array_equal(r0,r1) and np.array_equal(m0,m1)):
  raise RuntimeError('Exact feature/mask equivalence FAILED N='+str(n))
 with torch.inference_mode():
  ref=x[:,-1:,:]
  old=(torch.from_numpy(np.ascontiguousarray(x-ref)).cuda(),
       torch.from_numpy(r0).cuda(),torch.from_numpy(m0).cuda(),
       torch.from_numpy(np.ascontiguousarray(ref[:,0,:])).cuda())
  fast=ad['prepare'](x)
  y0=ad['forward'](old)
  y1=ad['forward'](fast)
  error=float((y0-y1).abs().max().item())
  if error>1e-6:raise RuntimeError('GPU forward mismatch N='+str(n)+' maxdiff='+str(error))
  check.append({'N':n,'feature_exact':True,'mask_exact':True,'max_pred_abs_error':error,'K':int(y1.shape[1])})
out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
out.write_text(json.dumps({'status':'PASS','details':check},indent=2)+'\n',encoding='utf8')
print('PROPOSED_GPU_EQUIVALENCE_OK',check)
