"""
"""
import math
import copy
import numpy as np
import random
import logging

import torch

torch.manual_seed(1234)
random.seed(1234)

epsilon = 1e-4


class NDPPSampler(object):
    def __init__(self, num_threads=1, device=torch.device("cpu")):
        self.num_threads = num_threads
        self.device = device

    def generate_samples(self, num_samples, dpp=None, V=None, B=None, C=None, L=None):
        samples = []

        if L is None:
            if dpp.disable_nonsym_embeddings and V is None:
                V = dpp.forward(dpp.all_items_in_catalog_set_var)
            elif V is None and B is None and C is None:
                V, B, D = dpp.forward(dpp.all_items_in_catalog_set_var)

            L = V.mm(V.transpose(0, 1))

            if not dpp.disable_nonsym_embeddings:
                C = D - D.transpose(0, 1)
                nonsymm = B.mm(C).mm(B.transpose(0, 1))
                kernel = L + nonsymm
            else:
                kernel = L
        else:
            kernel = L

        num_items_in_catalog = kernel.size()[0]

        eye = torch.eye(num_items_in_catalog)
        K = eye - (kernel + eye).inverse()

        for i in range(num_samples):
            K_copy = K.clone().detach()
            sample = []

            for j in range(num_items_in_catalog):
                if torch.rand(1) < K_copy[j, j]:
                    sample.append(j)
                else:
                    K_copy[j, j] -= 1

                K_copy[j + 1:, j] /= K_copy[j, j]
                K_copy[j + 1:, j + 1:] -= torch.ger(K_copy[j + 1:, j], K_copy[j, j + 1:])

            samples.append(sample)

        return samples

    def condition_dpp_on_items_observed(self, model, items_observed,
                                             V=None, B=None, C=None):

        all_items_in_catalog_set = model.all_items_in_catalog_set
        if V is None and B is None and C is None:
            if model.disable_nonsym_embeddings:
                V = model.forward(model.all_items_in_catalog_set_var)
                V = V.to(self.device)
            else:
                V, B, D = model.forward(model.all_items_in_catalog_set_var)
                V = V.to(self.device)
                B = B.to(self.device)
                C = D - D.transpose(0, 1)

        L = V.mm(V.transpose(0, 1))

        if not model.disable_nonsym_embeddings:
            nonsymm = B.mm(C).mm(B.transpose(0, 1))
            kernel = L + nonsymm
        else:
            kernel = L

        items_observed_set = set(items_observed)
        all_items_not_in_observed = list(all_items_in_catalog_set - items_observed_set)

        kernel_items_not_observed = kernel[all_items_not_in_observed, all_items_not_in_observed]
        kernel_items_observed = kernel[items_observed, :][:, items_observed]

        try:
            kernel_conditioned_on_items_observed = kernel_items_not_observed - \
                                                   kernel[all_items_not_in_observed, :][:, items_observed].mm(
                                                       kernel_items_observed.inverse()).mm(
                                                       kernel[items_observed, :][:, all_items_not_in_observed])
        except RuntimeError as e:
            eye = torch.eye(kernel_items_observed.size()[0]).to(self.device)
            kernel_items_observed += eye * epsilon
            kernel_conditioned_on_items_observed = kernel_items_not_observed - \
                                                   kernel[all_items_not_in_observed, :][:, items_observed].mm(
                                                       kernel_items_observed.inverse()).mm(
                                                       kernel[items_observed, :][:, all_items_not_in_observed])
        item_ids_to_K_conditioned_on_items_observed_row_col_indices = {}
        k_matrix_items_observed_row_index = 0
        for item_id in all_items_not_in_observed:
            item_ids_to_K_conditioned_on_items_observed_row_col_indices[item_id] = k_matrix_items_observed_row_index

            k_matrix_items_observed_row_index += 1

        return kernel_conditioned_on_items_observed, \
               item_ids_to_K_conditioned_on_items_observed_row_col_indices

    def condition_dpp_on_items_observed_greedy(self, model, items_observed, V=None, B=None, C=None):
        all_items_in_catalog_set = model.all_items_in_catalog_set
        if V is None and B is None and C is None:
            if model.disable_nonsym_embeddings:
                V = model.forward(model.all_items_in_catalog_set_var)
                V = V.to(self.device)
            else:
                V, _, D = model.forward(model.all_items_in_catalog_set_var)
                V = V.to(self.device)
                C = D - D.transpose(0, 1)
                C = C.to(self.device)

        if B is not None and torch.norm(V - B) > 0: 
            Z = torch.cat((V, B), axis=1)
            idx = np.arange(V.shape[1], V.shape[1]+B.shape[1])
            X = torch.eye(Z.shape[1]).to(C.device)
            X[np.ix_(idx,idx)] = C
            P = (Z @ X).detach()
            Q = Z.detach()
        else: 
            if model.disable_nonsym_embeddings: 
                P = V.detach().to(V.device)
            else:
                P = V.matmul(torch.eye(C.size(0)).to(C.device) + C).detach()
            Q = V.detach()

        items_observed_set = set(items_observed)
        all_items_not_in_observed = list(all_items_in_catalog_set - items_observed_set)

        num_items_in_catalog = len(all_items_in_catalog_set)
        num_items_observed = len(items_observed)

        marginal_gain = P.mul(Q).sum(1)
        observed_item = items_observed[0]
        C1 = torch.zeros(num_items_in_catalog, num_items_observed)
        C2 = torch.zeros(num_items_in_catalog, num_items_observed)
        for i in range(1, num_items_observed + 1):
            e1 = torch.matmul(Q, P[observed_item, :].reshape(-1, 1)).reshape(-1)
            e2 = torch.matmul(P, Q[observed_item, :].reshape(-1, 1)).reshape(-1)
            if i > 1:
                e1 -= torch.matmul(C1[:, :i - 1], C2[observed_item, :i - 1]).to(V.device)
                e2 -= torch.matmul(C2[:, :i - 1], C1[observed_item, :i - 1]).to(V.device)
            e1 /= marginal_gain[observed_item]
            C1[:, i - 1] = e1
            C2[:, i - 1] = e2

            marginal_gain -= e1.mul(e2).reshape(-1).to(V.device)
            if i >= num_items_observed:
                break
            observed_item = items_observed[i]
        conditional_prob = marginal_gain[all_items_not_in_observed]
        return conditional_prob

    def compute_next_item_probs_conditional_greedy(
            self, model, items_observed, next_items,
            return_next_item_probs_as_dict=True, V=None, B=None, C=None):

        next_item_probs_vec = self.condition_dpp_on_items_observed_greedy(
            model, items_observed, V=V, B=B, C=C)

        if return_next_item_probs_as_dict:
            next_item_probs = dict.fromkeys(next_items)
            next_item_probs_vec_index = 0
            for next_item_id in next_items:
                next_item_probs[next_item_id] = \
                    next_item_probs_vec[next_item_probs_vec_index]

                next_item_probs_vec_index += 1

            return next_item_probs
        else:
            return next_item_probs_vec

    def compute_next_item_probs_conditional_kdpp(
            self, model, items_observed, next_items,
            return_next_item_probs_as_dict=True, V=None, B=None, C=None):

        kernel_conditioned_on_items_observed, \
        item_ids_to_K_conditioned_on_items_observed_row_col_indices = self.condition_dpp_on_items_observed(
            model, items_observed, V=V, B=B, C=C)

        next_item_probs_vec = torch.empty(len(next_items))
        if return_next_item_probs_as_dict:
            next_item_probs = dict.fromkeys(next_items)
            for next_item_id in next_items:
                items_observed_row_col_index = item_ids_to_K_conditioned_on_items_observed_row_col_indices[next_item_id]

                next_item_probs[next_item_id] = kernel_conditioned_on_items_observed[items_observed_row_col_index,
                                                                                     items_observed_row_col_index]
                next_item_probs_vec = kernel_conditioned_on_items_observed[items_observed_row_col_index,
                                                                           items_observed_row_col_index]

            return next_item_probs
        else:
            return next_item_probs_vec, item_ids_to_K_conditioned_on_items_observed_row_col_indices

