import typing

import os

Header = os.path.dirname(os.path.abspath(__file__))
Header = Header + "/"

import csv
import logging

import pandas as pd
import numpy as np

from sklearn.utils import check_random_state

import torch
import torch.utils.data

from utils import VocabularyMapper


class BasketDataset(torch.utils.data.Dataset):
    nb_rows = None 

    def __init__(self,
                 orders=None,
                 product_catalog=None,
                 random_state=None,
                 min_basket_size=1,
                 max_basket_size=np.inf,
                 num_baskets=np.inf,
                 use_metadata=True,
                 map_product_ids=True,
                 use_fasttext=True,
                 columns_as_strings=None,
                 product_id_mapper=None,
                 get_negatives_associated_to_order_id=None,
                 filter_product_catalog=True):
        self.product_catalog = product_catalog
        self.orders = orders
        self.num_baskets = np.inf if num_baskets is None else num_baskets
        self.min_basket_size = min_basket_size
        self.max_basket_size = max_basket_size
        self.random_state = check_random_state(random_state)
        self.use_metadata = use_metadata
        self.map_product_ids = map_product_ids
        self.use_fasttext = use_fasttext
        self.columns_as_strings = columns_as_strings
        self.product_id_mapper = product_id_mapper
        self.get_negatives_associated_to_order_id = get_negatives_associated_to_order_id
        self.filter_product_catalog = filter_product_catalog
        self._load_data()

    def _listoflists_to_ordersdf(self, baskets):
        orders = []
        for order_id, basket in enumerate(baskets):
            orders += [(order_id, product_id) for product_id in basket]
        self.orders = pd.DataFrame(orders, columns=["order_id", "product_id"])
        return self.orders

    def clone(self):
        return BasketDataset(product_catalog=self.product_catalog,
                             use_metadata=self.use_metadata,
                             orders=self.orders,
                             map_product_ids=False,
                             columns_as_strings=self.columns_as_strings,
                             max_basket_size=self.max_basket_size,
                             get_negatives_associated_to_order_id=self.get_negatives_associated_to_order_id,
                             product_id_mapper=self.product_id_mapper,
                             filter_product_catalog=self.filter_product_catalog)

    def filter(self, order_ids):
        new = self.clone()
        new.orders = self.orders.loc[self.orders.order_id.isin(order_ids)]
        if self.filter_product_catalog:
            pids = new.orders.product_id.unique()
            new.product_catalog = self.product_catalog.loc[self.product_catalog.product_id.isin(pids)]
        new._prepare_baskets()
        return new

    def split(self, sizes: typing.List[int], shuffle: bool = True):
        datasets = []
        if np.sum(sizes) != len(self.baskets):
            raise ValueError("Sizes must sum to number of baskets")
        order_ids = self.orders.order_id.unique()
        if shuffle:
            self.random_state.shuffle(order_ids)
        start = 0
        logging.info("Splitting dataset")
        logging.info("Creating train dataset of size %d" % (sizes[0]))
        logging.info("Creating val dataset of size %d" % (sizes[1]))
        logging.info("Creating test dataset of size %d" % (sizes[2]))
        for size in sizes:
            logging.info("Creating dataset of size %d" % (size,))
            assert size > 0
            stop = start + size
            datasets.append(self.filter(order_ids[start:stop]))
            logging.info("Dataset created")
            start = stop
        return tuple(datasets)

    def _load_data(self):
        self._load_orders()
        self._prepare_baskets()
        self._load_product_catalog()
        if self.map_product_ids:
            self._map_product_ids()
        return self

    def read_csv(self, path):
        kwargs = {}
        if self.nb_rows is not None:
            kwargs['nb_rows'] = self.nb_rows
        if self.columns_as_strings is not None:
            dtype = {}
            for col in self.columns_as_strings:
                dtype[col] = str
            kwargs['dtype'] = dtype
        return pd.read_csv(path, **kwargs)

    def _load_product_catalog(self):
        if isinstance(self.product_catalog, str):
            product_catalog = self.read_csv(self.product_catalog)
            product_catalog = product_catalog.drop("Unnamed: 0", axis=1)
            if not self.use_metadata or not self.use_fasttext:
                fasttext_columns = []
                for col in product_catalog.columns:
                    if "fasttext" in col:
                        fasttext_columns.append(col)
                product_catalog = product_catalog.drop(fasttext_columns, axis=1)
            self.product_catalog = product_catalog
        elif self.product_catalog is None:
            self._load_orders()
            self.product_catalog = pd.DataFrame(self.orders["product_id"].unique(), columns=["product_id"])
        elif not isinstance(self.orders, pd.DataFrame):
            raise ValueError
        return self.product_catalog

    def _load_orders(self):
        if isinstance(self.orders, str):
            logging.info("Loading instacart orders from %s..." % (self.orders,))
            orders = self.read_csv(self.orders)
            cat = self.product_catalog.product_id.unique()
            self.orders = orders.loc[orders.product_id.isin(cat)]
            print(self.orders.head())
        elif not isinstance(self.orders, pd.DataFrame):
            self.orders = self._listoflists_to_ordersdf(self.orders)

        order_ids = self.orders.order_id.unique().tolist()
        if self.num_baskets < len(order_ids):
            logging.info("Only keeping %i baskets ..." % self.num_baskets)
            order_ids = self.random_state.choice(order_ids, size=self.num_baskets)
            self.orders = self.orders.loc[self.orders.order_id.isin(order_ids)]

    def _map_product_ids(self):
        common = self.product_catalog.product_id.unique()
        if self.filter_product_catalog:
            cat = self.product_catalog.product_id.unique()
            cat_in_orders = self.orders.product_id.unique()
            common = list(set(cat).intersection(cat_in_orders))
            self.orders = self.orders.loc[self.orders.product_id.isin(common)]
            self.product_catalog = self.product_catalog.loc[self.product_catalog.product_id.isin(common)]

        self.product_id_mapper = VocabularyMapper(common)
        self.orders.product_id = self._convert_product_ids(self.orders.product_id)
        self.product_catalog.product_id = self._convert_product_ids(self.product_catalog.product_id)

    def _convert_product_ids(self, product_ids):
        assert self.product_id_mapper, "_map_product_ids should be called first"
        return self.product_id_mapper(product_ids)

    def _prepare_baskets(self):
        grouped_items = self.orders[["order_id", "product_id"]].groupby("order_id")["product_id"].apply(list)
        order_ids = grouped_items.keys()
        baskets = pd.DataFrame(grouped_items)
        baskets["basket"] = baskets.pop("product_id")
        baskets["order_id"] = order_ids
        baskets["basket_size"] = baskets["basket"].apply(lambda basket: len(basket))
        baskets = baskets.loc[np.logical_and(baskets["basket_size"] <= self.max_basket_size,
                                             baskets["basket_size"] >= self.min_basket_size)]
        self.baskets = [np.unique(basket).tolist() for basket in baskets['basket'].tolist()]
        self.basket_ids = baskets["order_id"].tolist()
        self.negatives = self._compute_negatives()
        self.basket_sizes = baskets["basket_size"].tolist()
        # assert self.num_baskets == len(self.order_ids)
        self.num_baskets = len(self.baskets)
        logging.info("Dropping orders with invalid basket size")
        dropped_order_ids = [
            order_id for order_id, baskets in grouped_items.to_dict().items()
            if len(baskets) > self.max_basket_size or len(baskets) < self.min_basket_size
        ]
        self.orders = self.orders.drop(self.orders.index[self.orders.order_id.isin(dropped_order_ids)])
        logging.info("Baskets formed")
        logging.info("%i baskets" % len(baskets))

    def _compute_negatives(self):
        if self.get_negatives_associated_to_order_id is None:
            return None
        res = []
        for order_id in self.basket_ids:
            negatives = self.get_negatives_associated_to_order_id(order_id)
            res.append([self._convert_product_ids(neg) for neg in negatives])
        return res

    def __len__(self):
        return self.num_baskets

    def __getitem__(self, index):
        assert index < self.num_baskets
        return index

    def get_baskets(self, indices):
        return [self.baskets[i] for i in indices]

    def get_negatives(self, indices):
        if self.negatives is None:
            return
        return [self.negatives[i] for i in indices]

    def get_basket_size_buckets(self):
        basket_sizes_dict = {}
        max_basket_size = 0
        for basket in self.baskets:
            basket_size = len(basket)
            if basket_size > max_basket_size:
                max_basket_size = basket_size
            if basket_size not in basket_sizes_dict:
                basket_sizes_dict[basket_size] = 0
            basket_sizes_dict[basket_size] += 1
        self.basket_sizes = basket_sizes_dict

        max_basket_size = max(list(self.basket_sizes.keys()))
        basket_sizes = np.arange(1, max_basket_size + 1)
        boundaries = pd.cut(np.log10(basket_sizes), 3, right=False, labels=["small", "medium", "large"])
        self.buckets = dict(zip(basket_sizes, boundaries))
        return self.buckets


class BasketDataLoader(torch.utils.data.DataLoader):
    def __iter__(self):
        for indices in super(BasketDataLoader, self).__iter__():
            yield self.dataset.get_baskets(indices)

def load_ceil_dataset(input_file, **kwargs):
    orders = []
    logging.info('loading dataframe')
    data = pd.read_json(input_file)
    rank = kwargs["rank"]
    del kwargs["rank"]

    for i in range(data.shape[0]):
        num_ice = len(data.loc[i, "ctxs_candidates"])
        for j in range(int(rank*num_ice)):
            num_cand = len(data.loc[i, "ctxs_candidates"][0])
            for k in range(int(num_cand)):
                orders += [(data.loc[i, "idx"] * num_ice * num_cand + k + 1, data.loc[i, "ctxs_candidates"][j][k])]

    orders = pd.DataFrame(orders, columns=["order_id", "product_id"])
    logging.info(orders)
    return BasketDataset(orders=orders, **kwargs)


def load_dataset(dataset_name, **kwargs):
    logging.info("Loading %s dataset" % dataset_name)
    if dataset_name.lower() == 'geoquery' or dataset_name.lower() == 'nl2bash' or dataset_name.lower() == 'webqs' or dataset_name.lower() == 'mtop' or dataset_name.lower() == 'rocending':
        return load_ceil_dataset(**kwargs)
    else:
        raise NotImplementedError(dataset_name)
