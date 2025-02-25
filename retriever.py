import json
import logging
import faiss
import hydra
import hydra.utils as hu
import numpy as np
import torch
import tqdm
import os
from transformers import set_seed
from torch.utils.data import DataLoader
from utils.ndpp_map import greedy_map_ndpp
from utils.mcmc_utils import load_ndpp_kernel
from utils.misc import parallel_run, partial
from utils.collators import DataCollatorWithPaddingAndCuda
from models.biencoder import BiEncoder

logger = logging.getLogger(__name__)


class DenseRetriever:
    def __init__(self, cfg) -> None:
        self.cuda_device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.task_name = cfg.task_name
        self.dataset_reader = hu.instantiate(cfg.dataset_reader)
        co = DataCollatorWithPaddingAndCuda(tokenizer=self.dataset_reader.tokenizer, device=self.cuda_device)
        self.dataloader = DataLoader(self.dataset_reader, batch_size=cfg.batch_size, collate_fn=co)

        model_config = hu.instantiate(cfg.model_config)
        if cfg.pretrained_model_path is not None:
            self.model = BiEncoder.from_pretrained(cfg.pretrained_model_path, config=model_config)
        else:
            self.model = BiEncoder(model_config)

        self.model = self.model.to(self.cuda_device)
        self.model.eval()

        self.output_file = cfg.output_file
        self.num_candidates = cfg.num_candidates
        self.num_ice = cfg.num_ice
        self.is_train = cfg.dataset_reader.dataset_split == "train"

        self.ndpp_search = cfg.ndpp_search
        self.ndpp_topk = cfg.ndpp_topk
        self.mode = cfg.mode
        self.index = self.create_index(cfg)

    def create_index(self, cfg):
        logger.info("Building faiss index...")
        index_reader = hu.instantiate(cfg.index_reader)
        co = DataCollatorWithPaddingAndCuda(tokenizer=index_reader.tokenizer, device=self.cuda_device)
        dataloader = DataLoader(index_reader, batch_size=cfg.batch_size, collate_fn=co)

        index = faiss.IndexIDMap(faiss.IndexFlatIP(768))
        res_list = self.forward(dataloader, encode_ctx=True)
        id_list = np.array([res['metadata']['id'] for res in res_list])
        embed_list = np.stack([res['embed'] for res in res_list])
        index.add_with_ids(embed_list, id_list)
        faiss.write_index(index, cfg.faiss_index)
        logger.info(f"Saving faiss index to {cfg.faiss_index}, size {len(index_reader)}")
        # logger.info(id_list)
        return index

    def forward(self, dataloader, **kwargs):
        res_list = []
        for i, entry in enumerate(tqdm.tqdm(dataloader)):
            with torch.no_grad():
                metadata = entry.pop("metadata")
                res = self.model.encode(**entry, **kwargs)
            res = res.cpu().detach().numpy()
            res_list.extend([{"embed": r, "metadata": m} for r, m in zip(res, metadata)])
        return res_list

    def find(self):
        res_list = self.forward(self.dataloader)
        for res in res_list:
            res['entry'] = self.dataset_reader.dataset_wrapper[res['metadata']['id']]

        if self.ndpp_search:
            func = partial(ndpp, task_name=self.task_name, num_candidates=self.num_candidates, num_ice=self.num_ice,
                           mode=self.mode, ndpp_topk=self.ndpp_topk, scale_factor=self.model.scale_factor)
        else:
            func = partial(knn, num_candidates=self.num_candidates, num_ice=self.num_ice)
        data = parallel_run(func=func, args_list=res_list, initializer=set_global_object,
                            initargs=(self.index, self.is_train))

        with open(self.output_file, "w") as f:
            json.dump(data, f)


def set_global_object(index, is_train):
    global index_global, is_train_global
    index_global = index
    is_train_global = is_train


def knn(entry, num_candidates=1, num_ice=1):
    embed = np.expand_dims(entry['embed'], axis=0)
    near_ids = index_global.search(embed, max(num_candidates, num_ice)+1)[1][0].tolist()
    near_ids = near_ids[1:] if is_train_global else near_ids

    entry = entry['entry']
    entry['ctxs'] = near_ids[:num_ice]
    entry['ctxs_candidates'] = [[i] for i in near_ids[:num_candidates]]
    return entry

def get_ndpp_kernel(dataset, embed, candidates, scale_factor):
    V, B, C = load_ndpp_kernel(dataset, ondpp=False)
    candidates = [number for number in candidates if number < V.shape[0]]
    V = V[candidates, :]
    B = B[candidates, :]
    kernel_matrix = np.matmul(V, V.T) + np.matmul(np.matmul(B, C), B.T)
    kernel_matrix = kernel_matrix.numpy()
    near_reps = np.stack([index_global.index.reconstruct(i) for i in candidates], axis=0)
    embed = embed / np.linalg.norm(embed)
    near_reps = near_reps / np.linalg.norm(near_reps, keepdims=True, axis=1)

    rel_scores = np.matmul(embed, near_reps.T)[0]
    rel_scores = (rel_scores + 1) / 2
    rel_scores -= rel_scores.max()
    rel_scores = np.exp(rel_scores / (2 * scale_factor))
    kernel_matrix = rel_scores[None] * kernel_matrix * rel_scores[:, None]
    return near_reps, rel_scores, kernel_matrix, V, B, C

def random_sampling(num_total, num_ice, num_candidates, pre_results=None):
    ctxs_candidates_idx = [] if pre_results is None else pre_results
    while len(ctxs_candidates_idx) < num_candidates:
        samples_ids = np.random.choice(num_total, num_ice, replace=False).tolist()
        samples_ids = sorted(samples_ids)
        if samples_ids not in ctxs_candidates_idx:
            ctxs_candidates_idx.append(samples_ids)
    return ctxs_candidates_idx


def ndpp(entry, task_name='mrpc', num_candidates=1, num_ice=1, mode="map", ndpp_topk=100, scale_factor=0.1):
    candidates = knn(entry, num_ice=ndpp_topk)['ctxs']
    embed = np.expand_dims(entry['embed'], axis=0)
    near_reps, rel_scores, kernel_matrix, V, B, C = get_ndpp_kernel(task_name, embed, candidates, scale_factor)

    if mode == "cand_random" or np.isinf(kernel_matrix).any() or np.isnan(kernel_matrix).any():
        if np.isinf(kernel_matrix).any() or np.isnan(kernel_matrix).any():
            logging.info("Inf or NaN detected in Kernal_matrix, using random sampling instead!")
        topk_results = list(range(num_ice))
        ctxs_candidates_idx = [topk_results]
        ctxs_candidates_idx = random_sampling(num_total=ndpp_topk,  num_ice=num_ice,
                                              num_candidates=num_candidates,
                                              pre_results=ctxs_candidates_idx)
    elif mode == "pure_random":
        ctxs_candidates_idx = [candidates[:num_ice]]
        ctxs_candidates_idx = random_sampling(num_total=index_global.ntotal,  num_ice=num_ice,
                                              num_candidates=num_candidates,
                                              pre_results=ctxs_candidates_idx)
        entry = entry['entry']
        entry['ctxs'] = ctxs_candidates_idx[0]
        entry['ctxs_candidates'] = ctxs_candidates_idx
        return entry
    else:
        # MAP
        B = B.numpy()
        B = rel_scores[:, None] * B 
        B = torch.as_tensor(B)
        map_results = greedy_map_ndpp(B, C, num_ice)
        map_results = sorted(map_results)
        ctxs_candidates_idx = [map_results]
        #logger.info(f"ctxs_candidates_idx: {ctxs_candidates_idx}")

    ctxs_candidates = []
    for ctxs_idx in ctxs_candidates_idx:
        ctxs_candidates.append([candidates[i] for i in ctxs_idx])
    assert len(ctxs_candidates) == num_candidates

    entry = entry['entry']
    entry['ctxs'] = ctxs_candidates[0]
    entry['ctxs_candidates'] = ctxs_candidates
    return entry


@hydra.main(config_path="configs", config_name="retriever")
def main(cfg):
    logger.info(cfg)
    set_seed(43)
    retriever = DenseRetriever(cfg)
    retriever.find()


if __name__ == "__main__":
    main()
