import time
import torch
import numpy as np
from tqdm import tqdm
from src.utils.spectral import spectral_symmetrization
from src.utils.mcmc_utils import psd_matrix_sqrt,load_ndpp_kernel
from src.utils.tree_based_sampling import kdpp_tree_sampling_customized, construct_tree, construct_tree_fat_leaves
import logging

logger = logging.getLogger(__name__)

def kndpp_mcmc(tree, X, W, k, num_walks, rng):
    n = X.shape[0]
    assert k >= 2 and k <= n
    if rng is None:
        rng = np.random.RandomState(None)
    S = rng.permutation(n)[:k]
    num_rejections = []

    for _ in tqdm(range(num_walks), desc='Up-Down Random Walk'):
        T = rng.choice(S, k-2, replace=False)

        Xdown = X[T,:]
        W_cond = W - W @ Xdown.T @ ((Xdown @ W @ Xdown.T).inverse()) @ Xdown @ W
        W_cond_hat = (W_cond + W_cond.T)/2 + spectral_symmetrization((W_cond - W_cond.T)/2)
        What_sqrt = psd_matrix_sqrt(W_cond_hat)
        get_det_L = lambda S : (X[S,:] @ W_cond @ X[S,:].T).det()
        get_det_Lhat = lambda S : (X[S,:] @ W_cond_hat @ X[S,:].T).det()
        cnt = 0
        while(1):
            ab = kdpp_tree_sampling_customized(tree, What_sqrt, X, 2, rng)
            rand_num = rng.rand() if rng else np.random.rand()
            if rand_num < get_det_L(ab) / get_det_Lhat(ab):
                break
            cnt += 1
        num_rejections.append(cnt)
        S = np.union1d(T, ab)
    return S, num_rejections


def k_ndpp_sampling(X, W, k=10, random_state=1, ondpp=False, min_num_leaf=8, num_samples=10, pre_results=None):
    ctxs_candidates_idx = [] if pre_results is None else pre_results
    rng = np.random.RandomState(random_state)
    torch.random.manual_seed(random_state if random_state else rng.randint(99))

    n, d = X.shape

    # Preprocessing - tree construction
    tic = time.time()
    #print("[MCMC] Tree construction")
    if n >= 1e5:
        tree = construct_tree_fat_leaves(np.arange(n), X.T, min_num_leaf)
    else:
        tree = construct_tree(np.arange(n), X.T)
    time_tree_mcmc = time.time() - tic
    #print(f"[MCMC] tree construction time: {time_tree_mcmc:.5f} sec")

    # Set the mixing time to k^2
    num_walks = k ** 2
    for i in range(num_samples):
        tic = time.time()
        sample, num_rejects = kndpp_mcmc(tree, X, W, k, num_walks, rng)
        time_sample = time.time() - tic
        #print(f"{i} sample : {sample}")
        #print(f"{i} ctxs_candidates_idx : {ctxs_candidates_idx}")
        if sample not in ctxs_candidates_idx:
            assert len(sample) == k
            ctxs_candidates_idx.append(list(sample))
        #print(f"[MCMC] sampling time : {time_sample:.5f} sec")
        #print(f"[MCMC] num_rejections: {np.mean(num_rejects)}")

    logger.info(f"ctxs_candidates_idx: {ctxs_candidates_idx}")
    return ctxs_candidates_idx
