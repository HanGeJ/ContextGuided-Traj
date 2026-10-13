"""Inference adapters on ONE identical physical observation window.

Each existing checkpoint remains untouched. Never synthesizes 'predictions' or
falls back to a randomly initialized model. Every returned output is a genuine
K=20 forward tensor from an official/reproduced trained implementation.
"""
from __future__ import annotations
import os,sys,pickle,importlib,importlib.util
from pathlib import Path
import numpy as np


def load_torch(torch,path):
 try:return torch.load(path,map_location='cpu',weights_only=True)
 except Exception as e:
  if 'Weights only load failed' not in str(e):raise
  # These are user-owned checkpoints from previously trained project, not external untrusted files.
  return torch.load(path,map_location='cpu',weights_only=False)


def proposed(root,fold='univ',seed=42):
 import torch
 root=Path(root)
 ck=root/f'runs/tmm_stage3e_ethucy_final/ethucy_5fold_social/multihyp_social/{fold}/seed_{seed}/best_model.pt'
 if not ck.exists():raise FileNotFoundError(ck)
 obj=load_torch(torch,ck)
 if obj['args'].get('variant')!='multihyp_social' or int(obj['args'].get('num_modes',0))!=20:
  raise RuntimeError('Unexpected proposed model variant or K')
 cfg=obj['args'];sys.path.insert(0,str(root))
 from models.factory import build_model
 keys=('obs_len','pred_len','channels','individual','num_modes','hyp_hidden_dim','social_hidden_dim','social_dim')
 model=build_model('multihyp_social',**{k:cfg[k] for k in keys}).cuda().eval()
 model.load_state_dict(obj['model_state_dict'],strict=True)
 # Use the *same neighbor slots as the stored Stage3E training data*.
 npz=root/f'data/ethucy_5fold_social/{fold}/train.npz'
 with np.load(npz,allow_pickle=False) as f:
  if 'neighbor_rel' not in f:raise ValueError('Training dataset lacks neighbor topology, cannot certify matched adapter')
  topk=int(f['neighbor_rel'].shape[2])
 def prepare(obs):
  import torch
  n=obs.shape[0];ref=obs[:,-1:,:]
  centered=obs-ref
  # Exact-order vectorized replacement for v1.3's Python per-agent/per-timestep sort.
  # CPU equivalence is checked by tests/test_neighbor_equivalence.py before benchmarking.
  from neighbor_fast import neighbor_rel_vectorized
  r,m=neighbor_rel_vectorized(obs,topk)
  return (torch.from_numpy(np.ascontiguousarray(centered)).cuda(),
          torch.from_numpy(r).cuda(),torch.from_numpy(m).cuda(),
          torch.from_numpy(np.ascontiguousarray(ref[:,0,:])).cuda())
 def forward(v):
  obs,rel,mask,origin=v
  pred=model(obs,neighbor_rel=rel,neighbor_mask=mask)['candidates']
  return pred+origin[:,None,None,:] # native centered positions -> common global coords
 return dict(model=model,checkpoint=ck,prepare=prepare,forward=forward,expected='NKT2',
             notes=f'stage3e checkpoint strict load, temporal+social+K20, original per-time-step nearest social topk={topk}; geometry recovery included')


def tutr(root,fold='univ',seed=42):
 import torch
 root=Path(root);repo=root/'runs/tmm_efficiency_autobench_v1/workspace/repos/tutr'
 sys.path.insert(0,str(repo))
 from model import TrajectoryModel
 hpfile=repo/'config'/f'{fold}.py'
 spec=importlib.util.spec_from_file_location('tmm_v13_hp',str(hpfile));hp=importlib.util.module_from_spec(spec);spec.loader.exec_module(hp)
 ck=repo/'checkpoint'/fold/'best.pth'
 modepath=repo/'dataset'/f'{fold}_motion_modes.pkl'
 if not ck.exists() or not modepath.exists():raise FileNotFoundError(f'{ck} or {modepath}')
 with open(modepath,'rb') as f:modes=torch.as_tensor(pickle.load(f),dtype=torch.float32,device='cuda')
 if tuple(modes.shape[1:])!=(12,2) or modes.shape[0]<20:raise ValueError(f'Inadequate TUTR K20 modes shape={tuple(modes.shape)}')
 model=TrajectoryModel(in_size=2,obs_len=8,pred_len=12,embed_size=hp.model_hidden_dim,
                       enc_num_layers=2,int_num_layers_list=[1,1],heads=4,forward_expansion=2).cuda().eval()
 model.load_state_dict(load_torch(torch,ck),strict=True)
 cutoff=float(hp.dist_threshold)
 def prepare(obs):
  import torch
  n=len(obs);origin=obs[:,-1,:].copy()
  centered=obs-origin[:,None,:]
  angle=np.arctan2(centered[:,0,1],centered[:,0,0]);c=np.cos(angle);s=np.sin(angle)
  rot=np.stack([np.stack([c,-s],axis=-1),np.stack([s,c],axis=-1)],axis=-2).astype(np.float32)
  target=np.einsum('nti,nij->ntj',centered,rot).astype(np.float32)
  nearby=[];count=[]
  for i in range(n):
   d=np.linalg.norm(obs-obs[i:i+1],axis=-1).mean(axis=1)
   ids=np.flatnonzero(d < cutoff)
   if i not in ids:ids=np.r_[i,ids]  # native includes self
   ids=sorted(set(map(int,ids)),key=lambda j: (float(d[j]),j))[:50]
   neighborhood=np.einsum('nti,ij->ntj',(obs[ids]-origin[i]),rot[i]).astype(np.float32)
   nearby.append(neighborhood);count.append(len(ids))
  mx=max(count)
  neighbor=np.zeros((n,mx,8,2),dtype=np.float32)
  masks=np.zeros((n,mx,mx),dtype=np.int32)
  for i,xx in enumerate(nearby):
   nn=len(xx);neighbor[i,:nn]=xx;masks[i,:nn,:nn]=1
  return (torch.from_numpy(target).cuda(),torch.from_numpy(neighbor).cuda(),
          torch.from_numpy(masks).cuda(),
          torch.from_numpy(rot.copy()).cuda(),torch.from_numpy(origin.copy()).cuda())
 def forward(v):
  target,nei,mask,rot,origin=v
  pred,_=model(target,nei,modes,mask,None,test=True,num_k=20)
  if tuple(pred.shape)!=(target.shape[0],20,24):raise RuntimeError(f'Unexpected TUTR output {tuple(pred.shape)}')
  rel=pred.reshape(target.shape[0],20,12,2)
  return torch.matmul(rel,rot.transpose(1,2)[:,None,:,:])+origin[:,None,None,:]
 return dict(model=model,checkpoint=ck,prepare=prepare,forward=forward,expected='NKT2',
             notes=f'TUTR official train-time translation/rotation; obs-distance neighbor filter {cutoff}; outputs decoded into common coordinates')


def eigen(root,fold='univ',seed=42):
 import torch
 root=Path(root);repo=root/'runs/tmm_efficiency_autobench_v1/workspace/repos/eigen'
 if not (repo/'trainval.py').exists():raise FileNotFoundError(repo)
 os.chdir(repo);sys.path.insert(0,str(repo))
 import baseline
 from EigenTrajectory import EigenTrajectory
 from utils import DotDict,get_exp_config
 from utils.trainer import ETSGCNTrainer
 hp=get_exp_config(str(repo/'config'/f'eigentrajectory-sgcn-{fold}.json'))
 if int(hp.num_samples)!=20:raise ValueError(f'Eigen pretrained K={hp.num_samples}')
 class Args:pass
 args=Args();args.test=True;args.tag=f'TMMeff_sgcn_{fold}';args.gpu_id='0'
 bridge=baseline.sgcn
 hooks=DotDict({'model_forward_pre_hook':bridge.model_forward_pre_hook,
                'model_forward':bridge.model_forward,
                'model_forward_post_hook':bridge.model_forward_post_hook})
 trainer=ETSGCNTrainer(base_model=bridge.TrajectoryPredictor,model=EigenTrajectory,
                       hook_func=hooks,args=args,hyper_params=hp)
 ck=repo/'checkpoints'/args.tag/fold/'model_best.pth'
 if not ck.exists():raise FileNotFoundError(ck)
 trainer.load_model()
 model=trainer.model.cuda().eval()
 def prepare(obs):return torch.from_numpy(np.ascontiguousarray(obs.copy())).cuda()
 def forward(x):return model(x)['recon_traj']
 return dict(model=model,checkpoint=ck,prepare=prepare,forward=forward,expected='KNT2',
             notes='Complete trained EigenTrajectory descriptor+SGCN+anchor+reconstruction. Raw absolute observed coords as native loader provides.')


def vite(root,fold='univ',seed=42):
 import torch
 root=Path(root);repo=root/'runs/tmm_efficiency_autobench_v1/workspace/repos/vite'
 if fold!='univ':raise RuntimeError('ViTE released checkpoint only for UNIV')
 os.chdir(repo);sys.path.insert(0,str(repo))
 from utils.helper import load_config
 from model.ViTE_model import ViTE
 hp=load_config(str(repo/'configs/ethucy.yaml'))
 if int(hp.sample_k)!=20:raise ValueError(f'ViTE K={hp.sample_k} expected 20')
 ck=repo/'checkpoint/univ/univ_ckpt_best.pth'
 if not ck.exists():raise FileNotFoundError(ck)
 model=ViTE(hp).cuda().eval()
 obj=load_torch(torch,ck);model.load_state_dict(obj['state_dict'],strict=True)
 def prepare(obs):
  x=torch.from_numpy(np.ascontiguousarray(obs[None].copy())).cuda()
  xr=torch.zeros_like(x);xr[:,:,1:]=x[:,:,1:]-x[:,:,:-1];xr[:,:,0]=xr[:,:,1]
  return x,xr
 def forward(v):
  x,xr=v
  y,_=model(x,xr,log_moe_info=False)
  if hp.pred_rel:y=torch.cumsum(y,dim=3)+x[:,:,[-1]].unsqueeze(2)
  return y
 return dict(model=model,checkpoint=ck,prepare=prepare,forward=forward,expected='BNKT2',
             notes='Original ViTE UNIV weights, full native agent-graph forward + cumulative relative->absolute decoding')


def flow(root,fold='univ',seed=42):
 """Strict opt-in hook for a later **verified** full-K20 native FlowChain bridge.

 No sample copies, repeated K1 calls with identical noise, or density-update
 microtiming can be used in place of the native generator. Fail closed.
 """
 bridge=Path(root)/'runs/tmm_efficiency_autobench_v131/adapters/flow_full_k20.py'
 if bridge.exists():
  spec=importlib.util.spec_from_file_location('flow_full_k20_certified_v13',str(bridge))
  module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
  d=module.build(root,fold,seed)
  needed={'model','checkpoint','prepare','forward','expected','notes','native_sampling_k20_verified','official_sampler_source'}
  if not isinstance(d,dict) or not needed.issubset(d):raise RuntimeError('Custom verified FlowChain adapter contract incomplete')
  if d.get('native_sampling_k20_verified') is not True or not d.get('official_sampler_source'):
   raise RuntimeError('FlowChain adapter has no verifiable native full K20 sampler reference')
  if d['expected'] not in ('NKT2','KNT2','BNKT2'):raise RuntimeError('Unsupported FlowChain output layout')
  if not Path(d['checkpoint']).is_file():raise RuntimeError('FlowChain checkpoint absent')
  return d
 raise RuntimeError('FLOWCHAIN_K20_ADAPTER_PENDING: FlowChain v1.1 test reports nsample=1; no certified official full K20 canonical-scene adapter. The v1.3 probe collects native source-level evidence. No latency can be used until a verified bridge exists.')
