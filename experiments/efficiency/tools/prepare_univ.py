#!/usr/bin/env python3
"""Create one immutable real observed UNIV scene for ALL models.

 ViTE release has only UNIV weights, so only UNIV is eligible for a common-weights comparison.
 Uses only x[:,:8,:], never future ground truth or labels. Does NOT export trajectories in result ZIP.
"""
import argparse,os,sys,json,hashlib
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out',required=True)
p.add_argument('--index',type=int,default=0);p.add_argument('--min-agents',type=int,default=32);a=p.parse_args()
repo=Path(a.root)/'runs/tmm_efficiency_autobench_v1/workspace/repos/vite'
if not (repo/'utils/dataloader_eth.py').exists():raise RuntimeError('ViTE official repository is absent')
os.chdir(repo);sys.path.insert(0,str(repo))
import numpy as np
from utils.dataloader_eth import TrajectoryDataset
class DSArgs:dataset='univ'
ds=TrajectoryDataset(DSArgs(),str(repo/'datasets/ethucy/univ/test'),obs_len=8,pred_len=12,skip=1)
if a.index<0 or a.index>=len(ds):raise IndexError(f'UNIV scene index {a.index} out of {len(ds)}')
x,_=ds[a.index]
obs=x.cpu().detach().numpy().astype('<f4')
if obs.ndim!=3 or obs.shape[1:]!=(8,2):raise ValueError(f'Invalid shared scene: {obs.shape}')
if len(obs)<a.min_agents:raise RuntimeError(f'Scene index={a.index} has N={len(obs)} (<{a.min_agents}); set TMM_V13_SCENE_INDEX or reduce planned N')
if not np.isfinite(obs).all():raise ValueError('Nonfinite canonical observations')
obs=np.ascontiguousarray(obs)
out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
np.savez_compressed(out,obs=obs)
import hashlib
meta={'scene':'UNIV','dataset':'ETH-UCY','source':'official ViTE UNIV test scene only; no future ground truth used',
      'data_path':str(repo/'datasets/ethucy/univ/test'),'window_index':a.index,'agents_total':len(obs),
      'obs_steps':8,'future_steps':12,'available_agent_indices':list(range(len(obs))),
      'observation_sha256':hashlib.sha256(obs.tobytes()).hexdigest(),
      'positions_dtype':'float32','canonical_npz':str(out.resolve()),
      'policy':'identical source observations and agent subset across all eligible model adapters; native model-specific coordinate transformation permitted'}
Path(str(out)+'.manifest.json').write_text(json.dumps(meta,indent=2)+'\n')
print(json.dumps(meta,indent=2))
