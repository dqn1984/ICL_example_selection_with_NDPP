import os
import numpy as np
import torch
import argparse


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='recipe', type=str)
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--k', default=10, type=int)
    parser.add_argument('--num_samples', default=10, type=int)
    parser.add_argument('--min_num_leaf', default=8, type=int)
    parser.add_argument('--ondpp', default=False, type=bool)
    return parser.parse_args()


def psd_matrix_sqrt(A):
    eig_vals, eig_vecs = torch.linalg.eigh(A)
    idx = eig_vals > 1e-15
    return eig_vecs[:,idx] * eig_vals[idx].sqrt()


def load_ndpp_kernel(dataset, ondpp):
    
    if dataset == 'geoquery':
        file_path_ondpp = ''
        file_path = 'geoquery_sdim150_nsdim150_alpha0.1_rank0.1_VBC.torch'
    elif dataset == 'mtop':
        file_path_ondpp = ''
        file_path = 'mtop_cand100_sdim100_nsdim100_alpha0.1_rank1_VBC.torch'
    elif dataset == 'nl2bash':
        file_path_ondpp = ''
        file_path = 'nl2bash_sdim100_nsdim100_alpha0.1_rank1_VBC.torch'
    elif dataset == 'webqs':
        file_path_ondpp = ''
        file_path = 'webqs_sdim100_nsdim100_alpha0.1_rank1_VBC.torch'
    elif dataset == 'rocending':
        file_path_ondpp = ''
        file_path = 'rocending_sdim100_nsdim100_alpha0.1_rank1_VBC.torch'

    else:
        raise NotImplementedError

    V, B, D = saved_model['V'], saved_model['B'], saved_model['C']
    C = D - D.T

    return V, B ,C

def elementary_polynomial(k, eigen_vals):
    n = len(eigen_vals)
    E_poly = torch.zeros((k + 1, n + 1), dtype=eigen_vals.dtype)
    E_poly[0, :] = 1
    for l in range(1, k + 1):
        for n in range(1, n + 1):
            E_poly[l, n] = E_poly[l, n - 1] + eigen_vals[n - 1] * E_poly[l - 1, n - 1]
    return E_poly


def sample_kdpp_eigen_vecs(k, eig_vals, E_poly, rng=None):
    ind_selected = np.zeros(k, dtype=int)
    for n in range(len(eig_vals), 0, -1):
        rand_nums = rng.rand() if rng else np.random.rand()
        if rand_nums < eig_vals[n - 1] * E_poly[k - 1, n - 1] / E_poly[k, n]:
            k -= 1
            ind_selected[k] = n - 1
            if k == 0:
                break
    return ind_selected

