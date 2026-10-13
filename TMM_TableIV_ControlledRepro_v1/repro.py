#!/usr/bin/env python3
"""Controlled ETH-UCY augmentation ablation for an existing TMM-v1 repository.

Does not modify original training source or previously saved model checkpoints.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile
from collections import Counter

VERSION = '1.0'
PACKAGE = 'TMM_TableIV_ControlledRepro_v1'
FOLDS = ('eth','hotel','univ','zara1','zara2')
SEEDS = (42,43,44)
ARMS = {
    'base_losses': {'use_augment': False, 'rotation_max_deg':0.0, 'mask_prob':0.0, 'noise_std':0.0},
    'rotation_only': {'use_augment': True, 'rotation_max_deg':180.0, 'mask_prob':0.0, 'noise_std':0.0},
    'mask_noise_only': {'use_augment': True, 'rotation_max_deg':0.0, 'mask_prob':0.3, 'noise_std':0.02},
    'rotation_mask_noise': {'use_augment': True, 'rotation_max_deg':180.0, 'mask_prob':0.3, 'noise_std':0.02},
}
LABELS = {
    'base_losses':'Base losses', 'rotation_only':r'+ Rotation',
    'mask_noise_only':r'+ Mask/Noise', 'rotation_mask_noise':r'Rotation + Mask/Noise',
}
LEGACY = {'base_losses':(0.227,0.389), 'rotation_only':(0.213,0.368),
          'mask_noise_only':(0.218,0.377), 'rotation_mask_noise':(0.207,0.359)}
COMMON = {
    'variant':'multihyp_social', 'data_root':'data/ethucy_5fold_social',
    'val_source':'train_group', 'val_ratio':0.1, 'epochs':100,
    'batch_size':512, 'num_workers':8, 'lr':0.01,
    'best_k':20, 'num_modes':20, 'cls_lambda':0.2, 'nll_lambda':0.1,
    'ranking_loss':'hard_ce', 'checkpoint_metric':'minade_k',
    'eval_views':1, 'max_mask_steps':4, 'relative_coords':True,
}
TRACKED = (
    'config.py','train.py','models/components.py','models/factory.py',
    'models/__init__.py','data/dataset.py','data/__init__.py',
    'utils/metrics.py','utils/evaluation.py','utils/training.py',
    'utils/seed.py','utils/__init__.py',
)
BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
OUT = ROOT / 'runs' / 'tmm_tableIV_controlled_repro_v1'
DATA = ROOT / COMMON['data_root']


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha(path:Path):
    digest = hashlib.sha256()
    with path.open('rb') as f:
        while True:
            chunk=f.read(4*1024*1024)
            if not chunk: break
            digest.update(chunk)
    return digest.hexdigest()


def jsonload(path):
    try: return json.loads(path.read_text(encoding='utf-8'))
    except (ValueError, OSError, UnicodeError): return None


def atomic_json(path, data):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp = path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf8')
    tmp.replace(path)


def build_lock(fast=False):
    code={}
    for rel in TRACKED:
        p=ROOT/rel
        if not p.is_file(): raise RuntimeError(f'Missing source: {p}')
        code[rel]=sha(p)
    data={}
    for fold in FOLDS:
        for split in ('train','test'):
            p=DATA/fold/f'{split}.npz'
            if not p.is_file(): raise RuntimeError(f'Missing ETH-UCY data: {p}')
            data[f'{fold}/{split}.npz']= {'bytes':p.stat().st_size,'sha256':sha(p) if not fast else None}
    return {'package':PACKAGE,'version':VERSION,'common':COMMON,'arms':ARMS,
            'folds':FOLDS,'seeds':SEEDS,'source_hashes':code,'input_hashes':data,
            'training_script':'train.py',
            'macro_rule':'mean of 3 seed means within each held-out fold, then mean of 5 fold means',
            'checkpoint_selection':'train-group validation minade_k',
            'test_policy':'fixed protocol; evaluate held-out test after selecting checkpoint on train-group validation',
            'created_utc': utc()}


def check_dataset_groups():
    import numpy as np
    from data.dataset import _group_disjoint_split
    for fold in FOLDS:
        path=DATA/fold/'train.npz'
        for seed in SEEDS:
            pair=_group_disjoint_split(path,0.1,seed)
            if pair is None: raise RuntimeError(f'Group-disjoint validation unavailable: {fold} seed={seed}')
            a,b=pair
            if len(a)==0 or len(b)==0: raise RuntimeError(f'Empty train/val split: {fold} seed={seed}')
            if np.intersect1d(a,b).size: raise RuntimeError(f'Train/val index overlap: {fold} seed={seed}')
            print(f'[SPLIT] {fold:5} seed={seed}: train={len(a):5} validation={len(b):5}; group-disjoint OK',flush=True)


def check_source_semantics():
    text=(ROOT/'train.py').read_text(encoding='utf8')
    components=(ROOT/'models'/'components.py').read_text(encoding='utf8')
    needles = ('--use_augment','--no_augment','--rotation_max_deg','--noise_std',
               '--mask_prob','--max_mask_steps','--checkpoint_metric','--val_source',
               '--ranking_loss','--cls_lambda','--nll_lambda','--skip_test',
               '--num_modes','--eval_views','--max_train_samples')
    missing=[n for n in needles if n not in text]
    if missing: raise RuntimeError(f'Training script lacks required CLI flags: {missing}')
    if not all(x in components for x in ('def augment_batch_social(', 'if mask_prob > 0.0', 'if noise_std > 0.0', 'rotation_max_deg')):
        raise RuntimeError('Augmentation implementation differs from audited source; verify before running')
    if 'val_standard[metric_key] < best_value' not in text or 'if not args.skip_test:' not in text:
        raise RuntimeError('Checkpoint selection/test handling changed; refuse to run without review')
    print('[CODE] Audited augmentation switches, validation-only checkpoint selection, test gating: OK')


def check_env(require_cuda=True):
    import numpy
    import torch
    print(f'[ENV] python={sys.executable}; torch={torch.__version__}; numpy={numpy.__version__}')
    print(f'[ENV] CUDA={torch.cuda.is_available()}')
    if torch.cuda.is_available(): print(f'[ENV] device={torch.cuda.get_device_name(0)}')
    if require_cuda and not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable; choose correct Python/Conda environment, do not run 60 trainings on CPU')


def check(require_cuda=True,quick=False):
    os.chdir(ROOT)
    check_env(require_cuda)
    check_source_semantics()
    if not quick:
        check_dataset_groups()
        lock=build_lock()
        print('[LOCK] Code and data hashes generated', len(lock['source_hashes']), len(lock['input_hashes']))
    print('[CHECK] PASSED',flush=True)


def jobfolder(arm,fold,seed,outroot=OUT):
    return outroot/'jobs'/arm/'ethucy_5fold_social'/'multihyp_social'/fold/f'seed_{seed}'


def matches_run(d,arm,fold,seed,needs_test=True):
    if not isinstance(d,dict): return False,'missing summary'
    a=d.get('arguments',{})
    c=ARMS[arm]
    checks={
        'variant':'multihyp_social','seed':seed,'test_set':fold,
        'data_root':'data/ethucy_5fold_social','val_source':'train_group',
        'epochs':100,'batch_size':512,'lr':0.01,'best_k':20,'num_modes':20,
        'cls_lambda':0.2,'nll_lambda':0.1,'ranking_loss':'hard_ce',
        'eval_views':1, 'use_augment':c['use_augment'],
        'rotation_max_deg':c['rotation_max_deg'],
        'mask_prob':c['mask_prob'],'noise_std':c['noise_std'],
        'max_mask_steps':4, 'skip_test':not needs_test,
    }
    for k,v in checks.items():
        obs=a.get(k)
        if isinstance(v,(int,float)) and not isinstance(v,bool):
            if not isinstance(obs,(int,float)) or not math.isclose(obs,v,rel_tol=1e-7,abs_tol=1e-7):return False,f'{k}: {obs} != {v}'
        elif obs!=v:return False,f'{k}: {obs} != {v}'
    if d.get('checkpoint_metric')!='minade_k':return False,'checkpoint metric mismatch'
    if d.get('validation_source')!='train_group':return False,'validation source mismatch'
    if d.get('best_epoch') is None:return False,'no selected checkpoint'
    if needs_test:
        std=(d.get('test') or {}).get('standard') or {}
        if not all(isinstance(std.get(k),(int,float)) and math.isfinite(std[k])
                   for k in ('minade_k','minfde_k','top1_ade','top1_fde')):
            return False,'missing held-out metrics'
    else:
        if d.get('test') is not None:return False,'test was accessed in smoke mode'
    return True,'OK'


def command(arm,fold,seed,outroot=OUT,smoke=False):
    conf=ARMS[arm]
    argv=[sys.executable,'-u','train.py',
          '--variant','multihyp_social',
          '--data_root','data/ethucy_5fold_social',
          '--test_set',fold,'--val_source','train_group',
          '--val_ratio','0.1','--epochs',str(1 if smoke else COMMON['epochs']),
          '--batch_size',str(32 if smoke else COMMON['batch_size']),
          '--num_workers',str(0 if smoke else COMMON['num_workers']),
          '--seed',str(seed),'--lr',str(COMMON['lr']),
          '--best_k','20','--num_modes','20',
          '--cls_lambda','0.2','--nll_lambda','0.1',
          '--ranking_loss','hard_ce','--checkpoint_metric','minade_k',
          '--eval_views','1','--rotation_max_deg',str(conf['rotation_max_deg']),
          '--mask_prob',str(conf['mask_prob']),
          '--noise_std',str(conf['noise_std']),
          '--max_mask_steps','4','--out_dir',str(outroot/'jobs'/arm)]
    argv.append('--use_augment' if conf['use_augment'] else '--no_augment')
    if smoke:
        argv.extend(['--max_train_samples','64','--max_val_samples','64','--skip_test'])
    return argv


def run_one(arm,fold,seed,smoke=False):
    outroot=OUT/'smoke_only' if smoke else OUT
    target=jobfolder(arm,fold,seed,outroot)
    prior=jsonload(target/'summary.json')
    if not smoke and prior:
        valid,why=matches_run(prior,arm,fold,seed)
        if valid and (target/'best_model.pt').is_file() and (target/'history.json').is_file():
            print(f'[SKIP] verified completed {arm}/{fold}/seed{seed}',flush=True)
            return True
        raise RuntimeError(f'Existing output cannot be trusted: {target}: {why}; move aside manually, never overwrite')
    if smoke and prior:
        print(f'[SMOKE] Existing smoke output; skipped: {target}',flush=True)
        return True
    cmd=command(arm,fold,seed,outroot,smoke)
    log=outroot/'logs'/f'{arm}_{fold}_seed{seed}.log'
    log.parent.mkdir(parents=True,exist_ok=True)
    print(f'[START] {utc()} {arm} {fold} seed={seed}',flush=True)
    print('[CMD] '+' '.join(cmd),flush=True)
    with log.open('w',encoding='utf8') as fp:
        fp.write('[CMD] '+' '.join(cmd)+'\n')
        fp.flush()
        proc=subprocess.Popen(cmd,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
        for line in proc.stdout:
            print(line,end='',flush=True)
            fp.write(line)
        rc=proc.wait()
        fp.write(f'\n[RUN_RETURN_CODE] {rc}\n')
    if rc:
        print(f'[FAILED] {arm} {fold} seed={seed}; rc={rc}; log={log}',flush=True)
        return False
    status=jsonload(target/'summary.json')
    if smoke:
        if status is None or status.get('test') is not None:return False
    else:
        valid,why=matches_run(status,arm,fold,seed)
        if not valid:raise RuntimeError(f'Job returned rc=0 but output failed protocol verification: {target}: {why}')
    print(f'[DONE] {utc()} {arm}/{fold}/seed{seed}',flush=True)
    return True


def rows():
    output=[];bad=[]
    for arm in ARMS:
        for fold in FOLDS:
            for seed in SEEDS:
                p=jobfolder(arm,fold,seed)
                d=jsonload(p/'summary.json')
                valid,why=matches_run(d,arm,fold,seed)
                if valid and not (p/'best_model.pt').is_file():valid,why=False,'best_model.pt missing'
                if valid and not (p/'history.json').is_file():valid,why=False,'history.json missing'
                if not valid:
                    bad.append({'arm':arm,'fold':fold,'seed':seed,'path':str(p),'reason':why})
                    continue
                s=d['test']['standard']
                vs=d['selected_validation']['standard']
                output.append({'arm':arm,'fold':fold,'seed':seed,
                               'minade_k':float(s['minade_k']), 'minfde_k':float(s['minfde_k']),
                               'top1_ade':float(s['top1_ade']),'top1_fde':float(s['top1_fde']),
                               'val_minade_k':float(vs['minade_k']),
                               'val_minfde_k':float(vs['minfde_k']),
                               'best_epoch':d['best_epoch'],
                               'path':str(p/'summary.json')})
    return output,bad


def write_csv(path, r):
    if not r:return
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(r[0].keys()));w.writeheader();w.writerows(r)


def summarize():
    OUT.mkdir(parents=True,exist_ok=True)
    result,missing=rows()
    write_csv(OUT/'tableIV_runs.csv',result)
    grouped={};foldrows=[]
    metrics=('minade_k','minfde_k','top1_ade','top1_fde')
    for arm in ARMS:
        fr={}
        for fold in FOLDS:
            selected=[r for r in result if r['arm']==arm and r['fold']==fold]
            if len(selected)==3 and {r['seed'] for r in selected}==set(SEEDS):
                row={'arm':arm,'fold':fold,'n_seeds':len(selected)}
                for metric in metrics:row[metric]=sum(x[metric] for x in selected)/3
                foldrows.append(row);fr[fold]=row
        grouped[arm]={ 'complete_folds':list(fr.keys()),'folds':fr }
        if len(fr)==5:
            grouped[arm]['macro']={m:sum(fr[f][m] for f in FOLDS)/5 for m in metrics}
        else:
            grouped[arm]['macro']=None
    write_csv(OUT/'tableIV_byfold.csv',foldrows)
    complete=(len(result)==60 and len(missing)==0)
    pairs=[]
    if complete:
        for fold in FOLDS:
            for seed in SEEDS:
                x={arm:next(r for r in result if r['arm']==arm and r['fold']==fold and r['seed']==seed) for arm in ARMS}
                for arm in ARMS:
                    if arm=='base_losses':continue
                    pairs.append({'arm':arm,'fold':fold,'seed':seed,
                         'delta_minade_vs_base':x[arm]['minade_k']-x['base_losses']['minade_k'],
                         'delta_minfde_vs_base':x[arm]['minfde_k']-x['base_losses']['minfde_k'],
                         'delta_top1ade_vs_base':x[arm]['top1_ade']-x['base_losses']['top1_ade']})
        write_csv(OUT/'tableIV_paired_delta.csv',pairs)
    output={'version':VERSION,'status':'COMPLETE' if complete else 'PARTIAL',
            'total_required':60,'completed':len(result),'missing':missing,'variants':grouped,
            'legacy_table_for_context_only':{a:{'ade':v[0],'fde':v[1]} for a,v in LEGACY.items()},
            'generated_utc':utc(), 'original_table_values_are_not_assumed_valid':True}
    if complete:
        output['differences_from_original']={
            a:{'minade':grouped[a]['macro']['minade_k']-LEGACY[a][0],
               'minfde':grouped[a]['macro']['minfde_k']-LEGACY[a][1]}
            for a in ARMS}
    atomic_json(OUT/'tableIV_summary.json',output)
    lines=['# Table IV Controlled Reproduction', '',f'Generated: {utc()}','',
        f'**Status: {output["status"]}. Complete runs: {len(result)}/60.**','',
        'Architecture: multihyp_social; five ETH–UCY LOSO held-out folds × seeds 42,43,44.',
        'Train-group validation only; checkpoint selected by minADE@20; final held-out fold evaluated after selection.',
        'All runs trained for 100 epochs with Adam, learning rate 0.01 and batch size 512.',
        'Training-only augmentation switches differ between rows; code/dataset remain identical.',
        '', '| Arm | Complete fold/seed runs | minADE@20 | minFDE@20 |', '|---|---:|---:|---:|']
    for arm in ARMS:
        n=sum(1 for x in result if x['arm']==arm)
        mm=grouped[arm]['macro']
        lines.append(f'| {LABELS[arm]} | {n}/15 | {mm["minade_k"]:.4f} | {mm["minfde_k"]:.4f} |' if mm else f'| {LABELS[arm]} | {n}/15 | — | — |')
    if not complete:
        lines.extend(['','Table candidate has NOT been generated: complete all 60 jobs before using values in a paper.',
                      '', 'Missing or invalid records:']+[f'- {d["arm"]}/{d["fold"]}/seed{d["seed"]}: {d["reason"]}' for d in missing[:20]])
    else:
        lines.extend(['','## Paired differences relative to base losses',
                      '', 'Negative values indicate improved predictions.',
                      '', '| Arm | Δ minADE@20 | Δ minFDE@20 |', '|---|---:|---:|'])
        base=grouped['base_losses']['macro']
        for arm in ('rotation_only','mask_noise_only','rotation_mask_noise'):
            mm=grouped[arm]['macro']
            lines.append(f'| {LABELS[arm]} | {mm["minade_k"]-base["minade_k"]:+.4f} | {mm["minfde_k"]-base["minfde_k"]:+.4f} |')
        tex=[r'% Controlled experiment: only use after checking the provenance manifest.',
             r'\begin{table}[t]',r'\centering',
             r'\caption{Controlled robust-training ablation on ETH--UCY (meters; $K=20$). All configurations use the same trajectory--social architecture, with results averaged over five held-out scenes and three random seeds.}',
             r'\label{tab:controlled_robust_ablation}',
             r'\begin{tabular}{lcc}',r'\hline',
             r'Variant & minADE@20 & minFDE@20 \\',r'\hline']
        for arm in ARMS:
            mm=grouped[arm]['macro']
            tex.append(f'{LABELS[arm]} & {mm["minade_k"]:.3f} & {mm["minfde_k"]:.3f} '+r'\\')
        tex.extend([r'\hline',r'\end{tabular}',r'\end{table}'])
        (OUT/'TABLE_IV_REPRODUCTION_CANDIDATE.tex').write_text('\n'.join(tex)+'\n',encoding='utf8')
    (OUT/'tableIV_report.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print('\n'.join(lines[:25]),flush=True)
    return complete


def package():
    summarize()
    bundle=ROOT/'runs'/'TMM_TableIV_ControlledRepro_v1_review_bundle.zip'
    suffixes={'tableIV_summary.json','tableIV_runs.csv','tableIV_byfold.csv','tableIV_paired_delta.csv',
              'tableIV_report.md','experiment_lock.json','TABLE_IV_REPRODUCTION_CANDIDATE.tex'}
    bundle.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(bundle,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=8) as z:
        for path in sorted(OUT.iterdir()):
            if path.is_file() and path.name in suffixes:z.write(path,arcname=f'{PACKAGE}/{path.name}')
        for arm in ARMS:
            for fold in FOLDS:
                for seed in SEEDS:
                    p=jobfolder(arm,fold,seed)/'summary.json'
                    if p.is_file():z.write(p,arcname=f'{PACKAGE}/raw_summaries/{arm}/{fold}/seed_{seed}.json')
        for path in sorted((OUT/'logs').glob('*.log')) if (OUT/'logs').exists() else []:
            tail=''.join(path.read_text(encoding='utf8',errors='replace').splitlines(keepends=True)[-35:])
            z.writestr(f'{PACKAGE}/log_tails/{path.name}',tail)
    print(f'[PACKAGE] {bundle} ({bundle.stat().st_size} bytes)',flush=True)


def lock_provenance():
    expected=build_lock()
    stored=OUT/'experiment_lock.json'
    if stored.exists():
        prev=jsonload(stored)
        if not isinstance(prev,dict):raise RuntimeError('Corrupt lock: '+str(stored))
        for key in ('common','arms','folds','seeds','source_hashes','input_hashes','training_script'):
            if prev.get(key)!=expected.get(key):
                raise RuntimeError(f'Provenance changed at {key}. Stop. Do not mix results from different sources or data. Archive old runs first.')
        print('[LOCK] Existing code/data provenance matches',flush=True)
    else:
        atomic_json(stored,expected)
        print('[LOCK] Frozen code, data and configuration manifest created',flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['check','smoke','run','status','package'])
    parser.add_argument('--max-jobs',type=int,default=None,
        help='Optional execution cap; report stays PARTIAL. Default runs all 60.')
    args=parser.parse_args()
    os.chdir(ROOT)
    if args.action=='check':
        check(require_cuda=os.environ.get('ALLOW_CPU')!='1');return
    if args.action=='smoke':
        check(require_cuda=False,quick=False)
        for arm in ARMS:
            if not run_one(arm,'eth',42,smoke=True):raise SystemExit(3)
        print('[SMOKE] PASS (4/4 augmentation switches); no held-out test metrics produced');return
    if args.action=='status':
        summarize();return
    if args.action=='package':
        package();return
    if args.action=='run':
        import fcntl
        OUT.mkdir(parents=True,exist_ok=True)
        with (OUT/'process.lock').open('w') as lf:
            try:fcntl.flock(lf,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:raise RuntimeError('Another Table IV reproduction run is active')
            check(require_cuda=os.environ.get('ALLOW_CPU')!='1')
            lock_provenance()
            n=0
            try:
                for arm in ARMS:
                    for fold in FOLDS:
                        for seed in SEEDS:
                            if args.max_jobs is not None and n>=args.max_jobs:
                                print(f'[LIMIT] stopping at {n} checked jobs');return
                            if not run_one(arm,fold,seed):
                                raise RuntimeError(f'Failed at {arm}/{fold}/seed{seed}. Fix and re-run; completed jobs will be reused.')
                            n+=1
            finally:
                package()

if __name__=='__main__':
    try:main()
    except KeyboardInterrupt:
        print('[STOP] interrupted; completed jobs remain saved',file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print('[ERROR]',repr(e),file=sys.stderr)
        sys.exit(1)
