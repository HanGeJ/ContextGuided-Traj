#!/usr/bin/env python3
"""Fail-closed equivalence testing before performance comparison; no GPU necessary."""
from __future__ import annotations
import sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import numpy as np
from neighbor_fast import neighbor_rel_reference,neighbor_rel_vectorized

def test_one(obs,k):
    a,mask_a=neighbor_rel_reference(obs,k)
    b,mask_b=neighbor_rel_vectorized(obs,k)
    assert a.dtype==b.dtype==np.float32 and mask_a.dtype==mask_b.dtype==np.float32
    assert np.array_equal(a,b),f'Neighbor coordinates mismatch N={len(obs)},K={k}'
    assert np.array_equal(mask_a,mask_b),f'Neighbor mask mismatch N={len(obs)},K={k}'

rng=np.random.default_rng(20261013)
num=0
for n in (1,2,3,4,5,8,16,32,57):
    for k in (1,2,4,6):
        obs=rng.normal(size=(n,8,2)).astype(np.float32)
        test_one(obs,k);num+=1
        if n>=3:
            # Real exact-distance ties and duplicated trajectories.
            obs[2,:,:]=obs[1,:,:]
            test_one(obs,k);num+=1
        zeros=np.zeros((n,8,2),dtype=np.float32)
        test_one(zeros,k);num+=1
# Print only qualitative benchmark indication; not a measured AutoDL figure.
obs=rng.normal(size=(57,8,2)).astype(np.float32)
for fn in (neighbor_rel_reference,neighbor_rel_vectorized):
    vals=[]
    for _ in range(4):
        t=time.perf_counter();fn(obs,4);vals.append((time.perf_counter()-t)*1000)
    print(fn.__name__,'CPU ms median (local only):',float(np.median(vals)))
print('NEIGHBOR_EQUIVALENCE_OK',num,'cases')
