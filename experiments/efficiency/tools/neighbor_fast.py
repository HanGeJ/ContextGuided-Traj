"""Vectorized, exactly order-compatible (distance, agent-id) nearest-neighbor encoding.

Replacement for only the per-timestep Python loops in v1.3 Proposed.prepare.
Preserves target/subset, K, social slots, missing-neighbor masks and relative coordinates.
The original benchmark and original checkpoint are not modified.
"""
from __future__ import annotations
import numpy as np


def neighbor_rel_reference(obs: np.ndarray, topk: int):
    # Exact algorithm transcribed from v1.3 matched_adapters.proposed.
    n=obs.shape[0]
    r=np.zeros((n,8,topk,2),dtype=np.float32)
    m=np.zeros((n,8,topk),dtype=np.float32)
    for i in range(n):
        for t in range(8):
            ranked=sorted((j for j in range(n) if j!=i),
                          key=lambda j:(float(np.linalg.norm(obs[j,t]-obs[i,t])),j))[:topk]
            for k,j in enumerate(ranked):
                r[i,t,k,:]=obs[j,t]-obs[i,t]
                m[i,t,k]=1.0
    return r,m


def neighbor_rel_vectorized(obs: np.ndarray, topk: int):
    if obs.ndim!=3 or obs.shape[1:]!=(8,2) or obs.dtype!=np.float32:
        raise ValueError('Expected float32 observed trajectories of shape [N,8,2]')
    if not np.isfinite(obs).all():
        raise ValueError('Nonfinite observed positions')
    n=len(obs)
    if topk<1: raise ValueError('topk must be >=1')
    r=np.zeros((n,8,topk,2),dtype=np.float32)
    m=np.zeros((n,8,topk),dtype=np.float32)
    q=min(topk,n-1)
    if q==0:return r,m
    # [T,N,2], so each observed timestep has its own nearest-neighbor order.
    v=np.ascontiguousarray(obs.transpose(1,0,2))
    # np.linalg.norm(float32) exactly matches the reference norm on each pair.
    distances=np.linalg.norm(v[:,:,None,:]-v[:,None,:,:],axis=-1)
    diag=np.arange(n)
    distances[:,diag,diag]=np.inf
    # Agent indices already sorted 0..N-1, so stable tie-breaking matches (distance, j).
    indices=np.argsort(distances,axis=-1,kind='stable')[:,:,:q]
    rows=np.arange(8)[:,None,None]
    near=v[rows,indices]
    rel=near-v[:,:,None,:]
    r[:,:,:q,:]=rel.transpose(1,0,2,3)
    m[:,:,:q]=1.0
    return r,m
