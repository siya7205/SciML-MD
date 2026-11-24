# ============================================================== #
#  Module:      msm/bootstrap_msm.py
#  Description: Bootstrap uncertainty quantification for stationary distribution and transition matrix
#  Author:      Siya Jethliya
#  Copyright (c) 2026 SciVizAI — All rights reserved.
# ============================================================== #

#!/usr/bin/env python3
"""
Bootstrap MSM for uncertainty quantification.

Builds bootstrap samples of MSMs to compute confidence intervals
for stationary distributions, transition matrices, and MFPTs.
"""
import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
import yaml
from scipy import sparse

from deeptime.decomposition import TICA
from deeptime.clustering import KMeans
from deeptime.markov.msm import MaximumLikelihoodMSM



# -------------------------------------------------------------- #
# Function: load_config
# -------------------------------------------------------------- #
def load_config(config_path='configs/pipeline.yaml'):
    """Load configuration from YAML file."""
    config_path = Path(config_path)
    if not config_path.exists():
        return None
    with open(config_path) as f:
        return yaml.safe_load(f)



# -------------------------------------------------------------- #
# Function: bootstrap_resample
# -------------------------------------------------------------- #
def bootstrap_resample(X, method='frames', block_size=10, seed=None):
    """
    Create a bootstrap sample from trajectory.
    
    Args:
        X: Trajectory data (T x F)
        method: 'frames' for frame resampling, 'blocks' for block bootstrap
        block_size: Size of blocks for block bootstrap
        seed: Random seed
        
    Returns:
        X_boot: Resampled trajectory
    """
    rng = np.random.RandomState(seed)
    T = len(X)
    
    if method == 'frames':
        # Simple frame resampling
        indices = rng.choice(T, size=T, replace=True)
        return X[indices]
    
    elif method == 'blocks':
        # Block bootstrap to preserve temporal correlations
        n_blocks = T // block_size
        block_indices = rng.choice(n_blocks, size=n_blocks, replace=True)
        
        indices = []
        for b in block_indices:
            start = b * block_size
            end = min(start + block_size, T)
            indices.extend(range(start, end))
        
        # Pad or trim to match original length
        indices = indices[:T]
        return X[indices]
    
    else:
        raise ValueError(f"Unknown bootstrap method: {method}")



# -------------------------------------------------------------- #
# Function: fit_msm_pipeline
# -------------------------------------------------------------- #
def fit_msm_pipeline(X, lag_tica, dim_tica, n_clusters, lag_msm, seed_kmeans):
    """
    Fit complete MSM pipeline: TICA -> KMeans -> MSM.
    
    Args:
        X: Feature trajectory
        lag_tica: TICA lag time
        dim_tica: TICA dimensions
        n_clusters: Number of clusters
        lag_msm: MSM lag time
        seed_kmeans: Random seed for KMeans
        
    Returns:
        msm: Fitted MSM model
        dtraj: Discrete trajectory
    """
    # TICA
    tica = TICA(lagtime=lag_tica, dim=dim_tica).fit(X).fetch_model()
    Y = tica.transform(X)
    
    # KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, max_iter=100, n_jobs=1).fit(Y).fetch_model()
    dtraj = kmeans.transform(Y).astype(np.int64)
    
    # MSM
    msm = MaximumLikelihoodMSM(lagtime=lag_msm, reversible=True).fit(dtraj).fetch_model()
    
    return msm, dtraj



# -------------------------------------------------------------- #
# Function: compute_mfpts
# -------------------------------------------------------------- #
def compute_mfpts(msm):
    """
    Compute mean first passage times between all state pairs.
    
    Args:
        msm: Fitted MSM model
        
    Returns:
        mfpt_matrix: Matrix of MFPTs (n_states x n_states)
    """
    n_states = msm.n_states
    P = np.asarray(msm.transition_matrix, dtype=float)

    # Mean first-passage time i -> j, in MSM lag steps, from the fundamental
    # matrix Z = (I - P + 1 pi^T)^-1:   M[i, j] = (Z[j, j] - Z[i, j]) / pi[j].
    # One O(S^3) inverse for the whole matrix. The previous implementation took
    # an entry of a different inverse per (i, j) pair - O(S^5) - and returned
    # NEGATIVE values (it is not a first-passage time).
    mfpts = np.zeros((n_states, n_states))
    try:
        pi = np.asarray(msm.stationary_distribution, dtype=float)
        Z = np.linalg.inv(np.eye(n_states) - P + np.outer(np.ones(n_states), pi))
        mfpts = (np.diag(Z)[None, :] - Z) / pi[None, :]
        np.fill_diagonal(mfpts, 0.0)
        if not np.all(np.isfinite(mfpts)):
            raise np.linalg.LinAlgError("non-finite MFPT")
    except Exception:
        # Fall back to one linear solve per target state: (I - Q_j) m = 1.
        mfpts = np.zeros((n_states, n_states))
        for j in range(n_states):
            idx = np.array([k for k in range(n_states) if k != j])
            try:
                m = np.linalg.solve(np.eye(n_states - 1) - P[np.ix_(idx, idx)],
                                    np.ones(n_states - 1))
                mfpts[idx, j] = m
            except np.linalg.LinAlgError:
                mfpts[idx, j] = np.inf

    return mfpts



# -------------------------------------------------------------- #
# Function: _bootstrap_once
# -------------------------------------------------------------- #
def _bootstrap_once(i, X, method, block_size, seed, lag_tica, dim_tica,
                    n_clusters, lag_msm, seed_kmeans, n_states):
    """One bootstrap replicate. Independent of every other replicate (own seed),
    so replicates can run in any order or in parallel with identical results.

    Returns (pi, P, mfpt) padded to the reference size, None if the replicate
    has no states, or an Exception instance if it failed.
    """
    try:
        X_boot = bootstrap_resample(X, method=method, block_size=block_size,
                                    seed=seed + i)
        msm_boot, _ = fit_msm_pipeline(X_boot, lag_tica, dim_tica,
                                       n_clusters, lag_msm, seed_kmeans)
        n_boot = msm_boot.n_states
        if n_boot <= 0:
            return None
        pi_boot = np.zeros(n_states)
        pi_boot[:n_boot] = msm_boot.stationary_distribution
        P_boot = np.zeros((n_states, n_states))
        P_boot[:n_boot, :n_boot] = msm_boot.transition_matrix
        mfpt_padded = np.full((n_states, n_states), np.nan)
        mfpt_padded[:n_boot, :n_boot] = compute_mfpts(msm_boot)
        return pi_boot, P_boot, mfpt_padded
    except Exception as e:                      # reported by the caller
        return e



# -------------------------------------------------------------- #
# Function: bootstrap_msm
# -------------------------------------------------------------- #
def bootstrap_msm(features_path, output_dir, config_path=None, 
                  lag_tica=None, dim_tica=None, n_jobs=1):
    """
    Perform bootstrap MSM analysis with confidence intervals.
    
    Args:
        features_path: Path to features.npy
        output_dir: Output directory for models
        config_path: Path to config YAML
        lag_tica: TICA lag (uses config or best from selection if None)
        dim_tica: TICA dim (uses config or best from selection if None)
        n_jobs: parallel bootstrap workers (1 = serial; replicates are independent)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load configuration
    config = load_config(config_path) if config_path else load_config()
    if config is None:
        # Defaults
        n_bootstrap = 100
        method = 'frames'
        block_size = 10
        seed = 123
        n_clusters = 30
        lag_msm = 30
        seed_kmeans = 42
    else:
        n_bootstrap = config['bootstrap']['n_iterations']
        method = config['bootstrap']['method']
        block_size = config['bootstrap']['block_size']
        seed = config['seeds']['bootstrap']
        n_clusters = config['msm']['n_clusters']
        lag_msm = config['msm']['lag']
        seed_kmeans = config['seeds']['kmeans']
    
    # Load best VAMP-2 parameters if not provided
    if lag_tica is None or dim_tica is None:
        best_path = Path('reports/vamp2_best.json')
        if best_path.exists():
            with open(best_path) as f:
                best = json.load(f)
            lag_tica = best['lag']
            dim_tica = best['dim']
            print(f"Using VAMP-2 selected parameters: lag={lag_tica}, dim={dim_tica}")
        else:
            # Use config defaults
            lag_tica = config['tica']['default_lag'] if config else 10
            dim_tica = config['tica']['default_dim'] if config else 5
            print(f"Using default parameters: lag={lag_tica}, dim={dim_tica}")
    
    print(f"[1/4] Loading features from {features_path}")
    X = np.load(features_path)
    print(f"  Features shape: {X.shape}")
    
    print(f"[2/4] Fitting reference MSM")
    msm_ref, dtraj_ref = fit_msm_pipeline(X, lag_tica, dim_tica, n_clusters, lag_msm, seed_kmeans)
    n_states = msm_ref.n_states
    print(f"  Reference MSM: {n_states} states")
    
    print(f"[3/4] Bootstrap sampling ({n_bootstrap} iterations)")
    
    # Storage for bootstrap results
    pi_samples = []
    P_samples = []
    mfpt_samples = []
    
    args_common = (X, method, block_size, seed, lag_tica, dim_tica,
                   n_clusters, lag_msm, seed_kmeans, n_states)
    if n_jobs is not None and n_jobs != 1:
        from joblib import Parallel, delayed
        outcomes = Parallel(n_jobs=n_jobs)(
            delayed(_bootstrap_once)(i, *args_common) for i in range(n_bootstrap))
    else:
        outcomes = []
        for i in range(n_bootstrap):
            if (i + 1) % 10 == 0:
                print(f"  Iteration {i+1}/{n_bootstrap}")
            outcomes.append(_bootstrap_once(i, *args_common))

    for i, out in enumerate(outcomes):            # input order -> deterministic
        if isinstance(out, Exception):
            print(f"    Warning: Bootstrap {i} failed: {out}")
        elif out is not None:
            pi_samples.append(out[0])
            P_samples.append(out[1])
            mfpt_samples.append(out[2])
    
    print(f"  Successful bootstraps: {len(pi_samples)}/{n_bootstrap}")
    
    print(f"[4/4] Computing confidence intervals")
    
    # Compute CIs
    pi_samples = np.array(pi_samples)
    P_samples = np.array(P_samples)
    mfpt_samples = np.array(mfpt_samples)
    
    confidence_level = config['bootstrap']['confidence_level'] if config else 0.95
    alpha = 1 - confidence_level
    lower_percentile = 100 * alpha / 2
    upper_percentile = 100 * (1 - alpha / 2)
    
    # Stationary distribution CIs
    pi_df = pd.DataFrame({
        'state': range(n_states),
        'mean': pi_samples.mean(axis=0),
        'std': pi_samples.std(axis=0),
        'lower': np.percentile(pi_samples, lower_percentile, axis=0),
        'upper': np.percentile(pi_samples, upper_percentile, axis=0),
        'reference': msm_ref.stationary_distribution
    })
    
    pi_path = output_dir / 'pi_ci.parquet'
    pi_df.to_parquet(pi_path)
    print(f"  Saved π CIs to {pi_path}")
    
    # Transition matrix CIs (sparse storage)
    P_mean = P_samples.mean(axis=0)
    P_lower = np.percentile(P_samples, lower_percentile, axis=0)
    P_upper = np.percentile(P_samples, upper_percentile, axis=0)
    
    P_path = output_dir / 'P_ci.npz'
    np.savez_compressed(P_path,
                       mean=P_mean,
                       lower=P_lower,
                       upper=P_upper,
                       reference=msm_ref.transition_matrix,
                       n_states=n_states)
    print(f"  Saved P CIs to {P_path}")
    
    # MFPT CIs
    mfpt_mean = np.nanmean(mfpt_samples, axis=0)
    mfpt_lower = np.nanpercentile(mfpt_samples, lower_percentile, axis=0)
    mfpt_upper = np.nanpercentile(mfpt_samples, upper_percentile, axis=0)
    
    # Save as DataFrame for easier access
    mfpt_records = []
    for i in range(n_states):
        for j in range(n_states):
            if not np.isnan(mfpt_mean[i, j]):
                mfpt_records.append({
                    'from_state': i,
                    'to_state': j,
                    'mean': mfpt_mean[i, j],
                    'lower': mfpt_lower[i, j],
                    'upper': mfpt_upper[i, j]
                })
    
    mfpt_df = pd.DataFrame(mfpt_records)
    mfpt_path = output_dir / 'mfpt_ci.parquet'
    mfpt_df.to_parquet(mfpt_path)
    print(f"  Saved MFPT CIs to {mfpt_path}")
    
    # Save metadata
    metadata = {
        'n_bootstrap': len(pi_samples),
        'n_requested': n_bootstrap,
        'method': method,
        'block_size': block_size,
        'confidence_level': confidence_level,
        'n_states': n_states,
        'lag_tica': lag_tica,
        'dim_tica': dim_tica,
        'lag_msm': lag_msm,
        'n_clusters': n_clusters
    }
    
    meta_path = output_dir / 'bootstrap_metadata.json'
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved metadata to {meta_path}")



# -------------------------------------------------------------- #
# Function: main
# -------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(
        description='Bootstrap MSM for uncertainty quantification'
    )
    parser.add_argument('--features', required=True,
                       help='Path to features.npy')
    parser.add_argument('--output_dir', default='models/msm_bootstrap',
                       help='Output directory for bootstrap results')
    parser.add_argument('--config', default=None,
                       help='Path to pipeline.yaml config file')
    parser.add_argument('--lag_tica', type=int, default=None,
                       help='TICA lag time (uses VAMP-2 selection if not provided)')
    parser.add_argument('--dim_tica', type=int, default=None,
                       help='TICA dimensions (uses VAMP-2 selection if not provided)')
    parser.add_argument('--n_jobs', type=int, default=1,
                       help='Parallel bootstrap workers (1 = serial; -1 = all cores)')
    
    args = parser.parse_args()
    
    bootstrap_msm(args.features, args.output_dir, args.config,
                 args.lag_tica, args.dim_tica, n_jobs=args.n_jobs)


if __name__ == '__main__':
    main()
