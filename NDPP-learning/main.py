import os
import pandas as pd
import random
import numpy as np

import torch
from nonsymmetric_dpp_learning import compute_prediction_metrics
from utils import (logging, parse_cmdline_args)
from sklearn.utils import check_random_state
from results import Results
from featurizer import ProductCatalogEmbedder
from datasets import (load_dataset, BasketDataLoader)
from training import (L2Regularization,
                                        do_learning, eval_model, compute_log_likelihood)
from prediction import NDPPPrediction

class NDPP(NDPPPrediction):
    def __init__(self, product_catalog,
                 num_sym_embedding_dims=None, num_nonsym_embedding_dims=None,
                 features_setup={"product_id": {"num_sym_embedding_dims": 100,
                                                "num_nonsym_embedding_dims": 10}},
                 disable_gpu=False, epsilon=1e-5,
                 hidden_dims=None, activation="selu", logger=None,
                 random_state=None, dropout=None, noshare_v=False, 
                 ortho_v=False, **kwargs):
        super(NDPP, self).__init__(**kwargs)
        self.product_catalog = product_catalog
        self.num_items = len(product_catalog)
        self.num_sym_embedding_dims = num_sym_embedding_dims
        self.num_nonsym_embedding_dims = num_nonsym_embedding_dims
        self.features_setup = features_setup
        self.disable_gpu = disable_gpu
        self.epsilon = epsilon
        self.hidden_dims = hidden_dims
        self.activation = activation
        self.random_state = check_random_state(random_state)
        self.dropout = dropout
        self.noshare_v = noshare_v
        self.ortho_v = ortho_v
        assert not (not noshare_v and ortho_v)
        self._compile()

    def _compile(self):
        self.get_v_embeddings = ProductCatalogEmbedder(
            self.product_catalog, self.features_setup, self.num_sym_embedding_dims,
            activation=self.activation, hidden_dims=self.hidden_dims,
            dropout=self.dropout)
        
        if (self.num_nonsym_embedding_dims == 0):
            logging.info("num_nonsym_embedding_dims = 0; disabling non-symmetric components")
            self.disable_nonsym_embeddings = True
        else:
            if not self.noshare_v:
                self.get_b_embeddings = self.get_v_embeddings
            else:
                self.get_b_embeddings = ProductCatalogEmbedder(
                    self.product_catalog, self.features_setup, self.num_nonsym_embedding_dims,
                    activation=self.activation, hidden_dims=self.hidden_dims,
                    dropout=self.dropout)
            self.d_params = torch.nn.Parameter(torch.randn(
                self.num_nonsym_embedding_dims,
                self.num_nonsym_embedding_dims), requires_grad=True)

        # L2正则化
        self.reg = L2Regularization().regularization

        if not self.disable_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")
        logging.info("Using device: %s " % self.device)

        self.all_items_in_catalog = self.product_catalog.product_id.unique().tolist()
        self.all_items_in_catalog_set = set(self.all_items_in_catalog)
        self.item_catalog_size = len(self.all_items_in_catalog_set)
        self.all_items_in_catalog_set_var = torch.LongTensor(
            np.arange(self.item_catalog_size)).to(self.device)

    def forward(self, _):
        if self.disable_nonsym_embeddings:
            return self.get_v_embeddings().to(self.device)
        else:
            if not self.ortho_v:
                return self.get_v_embeddings().to(self.device), \
                    self.get_b_embeddings().to(self.device), \
                    self.d_params.to(self.device)
            else:
                V_ = self.get_v_embeddings().to(self.device)
                B_ = self.get_b_embeddings().to(self.device)
                return V_ - B_ @ torch.linalg.solve(B_.T @ B_, B_.T @ V_), \
                    B_, self.d_params.to(self.device)

    @staticmethod
    def compute_log_likelihood(model, baskets,
                               alpha_regularization=0.,
                               beta_regularization=0.,
                               gamma_regularization=0.,
                               reduce=True, checks=False, mapped=True):
        return compute_log_likelihood(model, baskets, 
                           alpha_regularization=alpha_regularization,
                           beta_regularization=beta_regularization,
                           gamma_regularization=gamma_regularization,
                           reduce=reduce, checks=checks, mapped=mapped)

    def compute_lambda_vec(self, all_items_with_replacement):
        all_items_with_replacement = torch.from_numpy(all_items_with_replacement)
        self.item_counts = torch.bincount(all_items_with_replacement)
        self.lambda_vec = torch.ones(len(self.product_catalog)).to(self.device)
        for i, count in enumerate(self.item_counts):
            if count == 0:
                self.lambda_vec[i] = 1.0
            else:
                self.lambda_vec[i] = 1. / count


def prepare_data(args, random_state=None, num_val_baskets=None,
                 num_test_baskets=None,
                 max_basket_size=np.inf):
    rng = check_random_state(random_state)
    ds = load_dataset(dataset_name=args.dataset_name,
                      num_baskets=args.num_baskets,
                      use_metadata=args.use_metadata,
                      random_state=rng,
                      rank=args.rank,
                      max_basket_size=max_basket_size,
                      input_file=args.input_file)

    # 分割训练/验证/测试集
    logging.info("Spliting dataset")
    num_train_baskets = len(ds.baskets) - num_val_baskets - num_test_baskets
    train_ds, val_ds, test_ds = ds.split([num_train_baskets, num_val_baskets,
                                          num_test_baskets])
    logging.info("%i train baskets" % len(train_ds.baskets))
    basket_sizes = [len(x) for x in train_ds.baskets]
    logging.info("%i val baskets" % len(val_ds.baskets))
    logging.info("%i test baskets" % len(test_ds.baskets))
    train_data_loader = BasketDataLoader(train_ds, batch_size=args.batch_size)

    return (ds.product_catalog, ds.get_basket_size_buckets(),
            train_data_loader, val_ds, test_ds)


class Args(object):

    @staticmethod
    def get_default_cli_args():
        return parse_cmdline_args()

    @staticmethod
    def build_from_cli():
        return Args(parse_cmdline_args())

    def __init__(self, args):
        self.args = args
        self.args_dict = vars(self.args)
        self.hidden_dims = self._compute_hidden_dims(self.args)
        self.lr = self._infer_learning_rate(self.args, self.hidden_dims)
        self.alpha = self._compute_alpha(self.args, self.hidden_dims)
        self.beta = self._compute_beta(self.args, self.hidden_dims)
        self.gamma = self.args.gamma
        self.rank = self.args.rank
        self.disable_eval = self.args_dict.pop("disable_eval")
        self.inference = self.args_dict.pop("inference")
        self.num_bootstraps = self.args_dict.pop("num_bootstraps")
        for param, value in self.args_dict.items():
            if value is not None:
                logging.info(".....args.%s: %s" % (param, value))

    def compute_features_setup(self, product_catalog):
        args = self.args
        features_setup = {
            "product_id": {"embedding_dim": args.product_id_embedding_dim}
        }

        if args.use_metadata:
            if "aisle_id" in product_catalog.columns:
                features_setup["aisle_id"] = {"embedding_dim": args.aisle_id_embedding_dim}
            if "department_id" in product_catalog.columns:
                features_setup["department_id"] = {"embedding_dim": args.department_id_embedding_dim}
        return features_setup

    @staticmethod
    def _compute_hidden_dims(args):
        hidden_dims = args.hidden_dims
        if hidden_dims is None:
            hidden_dims = []
        return hidden_dims

    @staticmethod
    def _infer_learning_rate(args, hidden_dims):
        logging.info("Hyper-parameters:")

        if args.learning_rate is None:
            if len(hidden_dims) < 2:
                lr = 0.1
            else:
                lr = 0.01
            logging.info(".....learning_rate: %g" % lr)
        else:
            lr = args.learning_rate
        return lr

    @staticmethod
    def _compute_alpha(args, hidden_dims):
        alpha = args.alpha
        if alpha is None:
            if len(hidden_dims) == 0:
                alpha = 1.
            else:
                alpha = 0.
            logging.info(".....alpha: %g" % alpha)
        return alpha

    @staticmethod
    def _compute_beta(args, hidden_dims):
        beta = args.beta
        if beta is None:
            if len(hidden_dims) == 0:
                beta = 1.
            else:
                beta = 0.
            logging.info(".....beta: %g" % beta)
        return beta

class Dataset(object):
    def __init__(self, args, seed, rng, num_val_baskets, num_test_baskets):
        (product_catalog, basket_size_buckets, train_data,
         val_data, test_data) = prepare_data(args, random_state=rng,
                                   num_val_baskets=num_val_baskets,
                                   num_test_baskets=num_test_baskets,
                                   max_basket_size=args.max_basket_size)
        self.seed = seed
        self.num_val_baskets = num_val_baskets
        self.num_test_baskets = num_test_baskets
        self.product_catalog = product_catalog
        self.basket_size_buckets = basket_size_buckets
        self.max_basket_size = max(self.basket_size_buckets.keys())
        self.train_data = train_data
        self.val_data = val_data.baskets
        self.test_data = test_data.baskets

class Experiment(object):

    @classmethod
    def build(cls, arguments, dataset):
        args = arguments.args
        logging.info("Building model for %s" % (args.scores_file,))
        model = cls._build_model_object(arguments, dataset.product_catalog,
                                        dataset.max_basket_size,
                                        dataset.seed)
        ofile = cls._load_model(arguments, model, dataset)
        return model, ofile

    @staticmethod
    def run(model, arguments, dataset, store_inference_scores=False):
        args = arguments.args
        args_dict = arguments.args_dict

        logging.info("Running inference on test data for %s" % (args.scores_file,))
        artifacts, _ = eval_model(model, dataset.val_data, inference=arguments.inference,
                                  test_data=dataset.test_data, end=True,
                                  buckets=dataset.basket_size_buckets,
                                  num_threads=args.num_threads,
                                  num_bootstraps=arguments.num_bootstraps)

        scores = artifacts["scores"]
        df = pd.DataFrame(scores)
        for param, value in args_dict.items():
            if param == "hidden_dims":
                value = ",".join(list(map(str, arguments.hidden_dims)))
            df[param] = value
        logging.info("Scores:")
        print(df)
        pid = os.getpid()
        logging.info("Process %i complete." % pid)
        return df

    @classmethod
    def _build_model_object(cls, arguments, product_catalog, max_basket_size,
                            seed):
        args = arguments.args
        model_cls = NDPP
        model_params = {param: getattr(args, param)
                        for param in ["hidden_dims",
                                      "activation",
                                      "disable_gpu",
                                      "dropout",
                                      "noshare_v",
                                      "ortho_v",
                                      "num_threads"]}
        model_params["num_sym_embedding_dims"] = cls._compute_num_sym_embeddings(args)
        model_params["num_nonsym_embedding_dims"] = cls._compute_num_nonsym_embeddings(args)
        features_setup = arguments.compute_features_setup(product_catalog)
        model = model_cls(product_catalog, features_setup=features_setup,
                          **model_params)

        if args.num_nonsym_embedding_dims == 0:
            model.disable_nonsym_embeddings = True
        else:
            model.disable_nonsym_embeddings = False

        logging.info("Built model:")
        print(model)
        return model

    @staticmethod
    def _compute_num_sym_embeddings(args):
        if args.num_sym_embedding_dims is None:
            num_sym_embedding_dims = 100
            if args.max_basket_size != np.inf:
                num_sym_embedding_dims = args.max_basket_size
            return num_sym_embedding_dims
        else:
            return args.num_sym_embedding_dims

    @staticmethod
    def _compute_num_nonsym_embeddings(args):
        if args.num_nonsym_embedding_dims is None:
            num_nonsym_embedding_dims = 10
            if args.num_sym_embedding_dims is not None:
                num_nonsym_embedding_dims = args.num_sym_embedding_dims / 10

            return num_nonsym_embedding_dims
        else:
            return args.num_nonsym_embedding_dims

    @classmethod
    def _load_model(cls, arguments, model, dataset):
        args_dict = arguments.args_dict
        loaded = None
        try:
            loaded = cls._load_serialized_model(arguments, model)
        except Exception as e:
            logging.error(f"Could not load serialized model due to '{e}'")
        ofile = None
        if loaded is None:
            logging.info("Couldn't load model checkpoint; will retrain")
            if model.is_baseline:
                return cls._learn_baseline_model(arguments, model, dataset)
            model.compute_lambda_vec(np.concatenate(dataset.train_data.dataset.baskets))
            ofile = cls._learn_dpp_model(arguments, model,
                                         dataset.train_data,
                                         dataset.val_data,
                                         dataset.test_data,
                                         dataset.basket_size_buckets)
            cls._serialize_model(arguments, model)
        else:
            logging.info("Loaded model from checkpoint")
        logging.info("Loaded model:")
        print(model)
        if model.num_nonsym_embedding_dims == 0:
            V = model.forward([]); V = V.detach().cpu()
            B, C = None, None
        else:
            V, B, C = model.forward([]); V = V.detach().cpu(); B = B.detach().cpu(); C = C.detach().cpu()
        torch.save({"V":V, "B":B, "C":C}, cls._get_persisted_model_path(arguments.args).replace(".torch", "_VBC.torch"))

        return ofile

    @classmethod
    def _load_serialized_model(cls, arguments, model):
        args = arguments.args
        if not cls._model_can_be_serialized(args):
            return
        persisted_models_path = cls._get_persisted_model_path(args)
        if os.path.exists(persisted_models_path):
            model.load_state_dict(torch.load(persisted_models_path))
            return model

    @classmethod
    def _serialize_model(cls, arguments, model):
        args = arguments.args
        if not cls._model_can_be_serialized(args):
            return

        persisted_model_path = cls._get_persisted_model_path(args)
        head, _ = os.path.split(persisted_model_path)
        if not os.path.exists(head):
            os.makedirs(head)
        torch.save(model.state_dict(), persisted_model_path)

    @staticmethod
    def _model_can_be_serialized(args):
        return args.persisted_model_dir is not None

    @classmethod
    def _get_persisted_model_path(cls, args):
        persisted_model_dir =args.persisted_model_dir
        fname =  cls._persisted_model_fname(args.scores_file).split('/')[-1]

        return os.path.join(persisted_model_dir, fname)

    @staticmethod
    def _persisted_model_fname(scores_file):
        return scores_file + ".torch"

    @staticmethod
    def _learn_baseline_model(arguments, model, dataset):
        return model.do_learning(dataset)

    @staticmethod
    def _learn_dpp_model(arguments, model, train_data, val_data,
                         test_data, basket_size_buckets):
        args_dict = arguments.args_dict
        args = arguments.args
        # 训练NDPP
        _, ofile = do_learning(model,
                               **{"train_data": train_data,
                                  "val_data": val_data,
                                  "test_data": test_data,
                                  "num_iterations": args.num_iterations,
                                  "alpha_train": arguments.alpha,
                                  "beta_train": arguments.beta,
                                  "gamma_train": arguments.gamma,
                                  "disable_eval": arguments.disable_eval,
                                  "inference": arguments.inference,
                                  "learning_rate": arguments.lr,
                                  "eval_freq": 20,
                                  "buckets": basket_size_buckets,
                                  "num_bootstraps": arguments.num_bootstraps,
                               })
        return ofile

class NewNDPP(NDPP):

    def __init__(self, **kwargs):
        super(OrthogonalNDPP, self).__init__(**kwargs)
        self.d_params = torch.nn.Parameter(torch.randn(self.num_nonsym_embedding_dims // 2))

    def get_sigmas(self):
        return torch.exp(self.d_params)

    def forward(self, _):
        V = self.get_v_embeddings().to(self.device)
        B = self.get_b_embeddings()
        B, _ = torch.qr(B)

        B = B.to(self.device)

        D = torch.zeros(self.num_nonsym_embedding_dims).to(self.device)
        D[::2] = self.get_sigmas()
        D = torch.diag_embed(
            D,
            offset=1,
        )[:-1, :-1]

        if self.ortho_v:
            V = V - B @ torch.linalg.solve(B.T @ B, B.T @ V)
            return V, B, D
        else:
            return V, B, D


class NewExperiment(Experiment):

    @classmethod
    def _build_model_object(cls, arguments, product_catalog, max_basket_size, seed):
        args = arguments.args
        model_cls = NewNDPP
        model_params = {
            param: getattr(args, param)
            for param in ["hidden_dims", "activation", "disable_gpu", "dropout", "noshare_v", "ortho_v", "num_threads"]
        }
        model_params["num_sym_embedding_dims"] = cls._compute_num_sym_embeddings(args)
        model_params["num_nonsym_embedding_dims"] = cls._compute_num_nonsym_embeddings(args)
        model_params["product_catalog"] = product_catalog
        features_setup = arguments.compute_features_setup(product_catalog)
        model_params["features_setup"] = features_setup
        model = model_cls(**model_params)

        if args.num_nonsym_embedding_dims == 0:
            model.disable_nonsym_embeddings = True
        else:
            model.disable_nonsym_embeddings = False

        logging.info("Built model:")
        print(model)
        return model


if __name__ == "__main__":
    arguments = Args.build_from_cli()
    args = arguments.args
    args.scores_file = args.scores_file.replace("scores", "scores-spectral")
    args_dict = arguments.args_dict
    num_val_baskets = args.num_val_baskets
    num_test_baskets = args.num_test_baskets
    seed = args.seed
    print(f"seed: {seed}")
    rng = check_random_state(seed)

    dataset = Dataset(args, seed, rng, num_val_baskets, num_test_baskets)

    model, ofile = NewExperiment.build(arguments, dataset)

    results_df = Experiment.run(model, arguments, dataset, store_inference_scores=True)
    res = Results(args.dataset_name, results_df)
    print(res)
