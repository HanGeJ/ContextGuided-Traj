#!/usr/bin/env python3
"""Read-only FlowChain native source inventory: no speculative K20 timings."""
import os,re,json,argparse,ast
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out',required=True);a=p.parse_args()
root=Path(a.root);repo=root/'runs/tmm_efficiency_autobench_v1/workspace/repos/flow'
output=Path(a.out);output.mkdir(parents=True,exist_ok=True)
report={'status':'FLOW_K20_NOT_YET_CERTIFIED','repo_found':repo.exists(),'version':'1.3',
        'reason':'v1.1 native test records nsample=1; missing certified full K20 callable accepting this exact canonical UNIV window',
        'files':[],'checkpoint_candidates':[],'recommended_next_step':'Inspect included source snippets to implement and validate official sample(K=20) before admitting FlowChain to same-N comparison.'}
patterns=('nsample','sample(','sample_','sample =','sample_size','num_samples','n_samples','predict','inference','evaluate')
if repo.exists():
 for f in sorted((repo/'src').rglob('*.py')):
  if not f.is_file() or f.stat().st_size>3_000_000:continue
  try:s=f.read_text(errors='replace')
  except OSError:continue
  ls=s.splitlines();hits=[]
  for k,line in enumerate(ls):
   if any(q in line.lower() for q in patterns):
    hits.append({'line':k+1,'context':'\n'.join(f'{j+1}: {ls[j]}' for j in range(max(0,k-2),min(len(ls),k+3)))[:850]})
  if hits:report['files'].append({'relative_path':str(f.relative_to(repo)),'size':f.stat().st_size,'hits':hits[:35],
                                  'total_hits':len(hits)})
 # output and checkpoint discovery, metadata only, no model tensors uploaded
 for p in sorted((repo/'output').rglob('*')) if (repo/'output').exists() else []:
  if p.is_file() and p.suffix.lower() in ('.pt','.pth','.ckpt','.pkl','.bin'):
   report['checkpoint_candidates'].append({'path':str(p.relative_to(repo)),'size_bytes':p.stat().st_size})
 for cfg in sorted((repo/'config/TP/FlowChain').rglob('*univ*.yml')) if (repo/'config/TP/FlowChain').exists() else []:
  report.setdefault('univ_config_summaries',[]).append({'path':str(cfg.relative_to(repo)),'content':cfg.read_text(errors='replace')[:14000]})
output.joinpath('FLOWCHAIN_SOURCE_AUDIT.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
print('FLOW_PROBE','files=',len(report['files']),'checkpoints=',len(report['checkpoint_candidates']))
