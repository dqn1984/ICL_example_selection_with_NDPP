import os
import logging
import time
from math import ceil

import numpy as np

from joblib import Parallel, delayed

import torch
from torch import multiprocessing as mp


from utils import LogLikelihood
from prediction import NDPPPrediction, get_auc_from_logliks



class L2Regularization(torch.autograd.Function):
    @staticmethod
    def regularization(V, B, C, lambda_vec, alpha, beta, gamma=0.0):
        V_regularization_term = 0
        B_regularization_term = 0

        if alpha != 0:
            V_norm = torch.norm(V, p=2, dim=1)
            V_regularization_term = alpha / 2.0 * lambda_vec.matmul(torch.pow(V_norm, 2))

            if B is not None and torch.norm(V - B) > 1e-10:
                B_norm = torch.norm(B, p=2, dim=1)
                B_regularization_term = beta / 2.0 * lambda_vec.matmul(torch.pow(B_norm, 2))

        return V_regularization_term + B_regularization_term 

class NDPP(NDPPPrediction):
    def __init__(self, num_threads=1):
        super(NDPPPrediction, self).__init__()


def compute_log_likelihood(model, baskets, alpha_regularization=0.,
                           beta_regularization=0., gamma_regularization=0.,
                           reduce=True, checks=False, mapped=True):
    return LogLikelihood.compute_log_likelihood(model,
                                                baskets,
                                                alpha_regularization=alpha_regularization,
                                                beta_regularization=beta_regularization,
                                                gamma_regularization=gamma_regularization,
                                                reduce=reduce,
                                                checks=checks,
                                                mapped=mapped)

def _compute_prediction_metrics_for_bootstrap(
        model, test_data, buckets, logger=None, num_test_baskets=10,
        random_state=None, iteration=np.nan, prefix="", negatives=None):
    from sklearn.utils import check_random_state
    rng = check_random_state(random_state)
    mpr = []
    prec_at_5 = []
    prec_at_10 = []
    scores = {"MPR": {}, "Prec@5": {}, "Prec@10": {}, "AUC": {}}

    test_data = list(enumerate(test_data[:]))
    if negatives is None:
        to_shuffle = test_data
    else:
        to_shuffle = list(zip(test_data, negatives))

    rng.shuffle(to_shuffle)
    if num_test_baskets >= len(test_data):
        logging.info("%i test baskets" % num_test_baskets)
        logging.info("%i test data" % len(test_data))
        raise ValueError
    dataset = to_shuffle[:num_test_baskets]
    test_data = dataset
    if negatives is not None:
        test_data = [x[0] for x in dataset]
        negatives = [x[1] for x in dataset]

    test_data = test_data[:num_test_baskets]
    test_data_idx, test_data = [x[0] for x in test_data], [x[1] for x in test_data]

    pos_pred_logliks, random_logliks = model.get_AUC_results_for_data(
        test_data, return_raw=True, negatives=negatives)
    test_auc = get_auc_from_logliks(pos_pred_logliks, random_logliks)

    for basket, pos, rand in zip(test_data, pos_pred_logliks,
                                 random_logliks):
        bucket = buckets[len(basket)]
        if bucket not in scores["AUC"]:
            scores["AUC"][bucket] = []
        scores["AUC"][bucket].append((pos, rand))

    if model.disable_nonsym_embeddings:
        V = model.forward(model.all_items_in_catalog_set_var)
        B, C = None, None
    else:
        V, B, D = model.forward(model.all_items_in_catalog_set_var)
        C = D - D.transpose(0, 1)

    for prediction, target, basket in model.get_predictions(
            test_data, V=V, B=B, C=C):
        pre_mpr = model._get_pre_mpr(prediction, target)
        mpr.append(pre_mpr)
        top5 = model._get_top_k(5, prediction, target)
        prec_at_5.append(top5)
        top10 = model._get_top_k(10, prediction, target)
        prec_at_10.append(top10)

        bucket = buckets[len(basket)]
        for name in scores:
            if bucket not in scores[name]:
                scores[name][bucket] = []
        scores["MPR"][bucket].append(pre_mpr)
        scores["Prec@5"][bucket].append(top5)
        scores["Prec@10"][bucket].append(top10)

    aux = {}
    for name, values in scores.items():
        aux[name] = {}
        for bucket, bucket_values in values.items():
            if len(bucket_values) == 0:
                continue

            if name == "AUC":
                pos_pred_logliks, random_logliks = zip(*bucket_values)
                agg = get_auc_from_logliks(pos_pred_logliks,
                                           random_logliks)
            else:
                agg = 100 * np.mean(bucket_values)

            logging.info("%s%s baskets %s at iteration %s: %g" % (
                prefix, bucket, name, iteration, agg))

            if logger:
                logger.add_scalar("%s/test-%s-baskets" % (name, bucket),
                agg, global_step=iteration)

            aux[name][bucket] = agg
    scores = aux

    if iteration is not None:
        mpr = 100 * np.mean(mpr)
        prec_at_5 = 100. * np.sum(prec_at_5) / float(len(prec_at_5))
        prec_at_10 = 100. * np.sum(prec_at_10) / float(len(prec_at_10))
        scores["MPR"]["all"] = mpr
        scores["Prec@5"]["all"] = prec_at_5
        scores["Prec@10"]["all"] = prec_at_10
        scores["AUC"]["all"] = test_auc
        logging.info("%sMPR for test at iteration %s : %s" % (prefix,
                                                              iteration,
                                                              mpr))
        logging.info("%sPrec@5 for test at iteration %s : %s" % (
            prefix, iteration, prec_at_5))
        logging.info("%sPrec@10 for test at iteration %s : %s" % (
            prefix, iteration, prec_at_10))
    else:
        for metric, values in scores.items():
            scores[metric]["all"] = np.mean(list(values.values()),
                                            axis=0)
    return scores

def compute_prediction_metrics(model, test_data, buckets, logger=None,
                               num_test_baskets=10, num_bootstraps=1,
                               iteration=np.nan, prefix="", num_threads=1,
                               negatives=None):
    scores = Parallel(n_jobs=num_threads)(delayed(
        _compute_prediction_metrics_for_bootstrap)(
            model, test_data, buckets, num_test_baskets=num_test_baskets,
            random_state=bidx, prefix=prefix, negatives=negatives,
            logger=logger if num_bootstraps == 1 else None,
            iteration=iteration if num_bootstraps == 1 else None)
            for bidx in range(num_bootstraps))

    df = []
    for bootstrap, stuff in enumerate(scores):
        aux = {"bootstrap": bootstrap}
        for metric, values in stuff.items():
            for bucket, value in values.items():
                aux["test-%s-baskets.%s" % (bucket, metric)] = value
        df.append(aux)
    scores = df

    if logger is None:
        return scores

    return scores

def eval_model(model, val_data, test_data=None, buckets=None, inference=False,
               env=None, end=False, num_bootstraps=1, num_threads=1, eval_freq=40,
               inference_freq=10, negative_val_data=None, negative_test_data=None):
    model.eval()

    inference = inference or end

    if env is None:
        env = {}
    logger = env.get("logger", None)
    prefix = env.get("prefix", "")
    iteration = env.get("iteration", -1)
    curr_auc = env.get("curr_auc", None)

    if inference or end or (iteration > 0 and iteration % 100 == 0) and test_data is not None:
        avg_test_log_likelihood = model.compute_log_likelihood(
            model, test_data, alpha_regularization=0, mapped=False)
        avg_test_log_likelihood = avg_test_log_likelihood.item()
        logging.info("%sAvg loglik for test at iteration %s: %g" % (
            prefix, iteration, avg_test_log_likelihood))

        avg_val_log_likelihood = model.compute_log_likelihood(
            model, val_data, alpha_regularization=0, mapped=False)
        avg_val_log_likelihood = avg_val_log_likelihood.item()
        logging.info("%sAvg loglik for val at iteration %s: %g" % (
            prefix, iteration, avg_val_log_likelihood))

    # 验证
    if not inference and not end and (iteration % eval_freq == 0):
        return locals(), False

    prev_auc = curr_auc

    curr_auc = model.get_AUC_results_for_data(val_data, negatives=negative_val_data)
    logging.info("%sAUC for val at iteration %s: %g" % (prefix, iteration,
                                                        curr_auc))
    if logger:
        logger.add_scalar("auc/val", curr_auc, global_step=iteration)

        for name, param in model.get_embeddings.named_parameters():
            logger.add_histogram(name, param, iteration)

    if inference and (end or iteration % inference_freq == 0):
        logging.info("Computing prediction metrics...")

        if negative_test_data is not None:
            negative_copy = negative_test_data[:]
        else:
            negative_copy = None

        scores = compute_prediction_metrics(model, test_data[:], buckets,
                                            logger=logger, iteration=iteration,
                                            num_threads=num_threads,
                                            num_bootstraps=num_bootstraps,
                                            prefix=prefix,
                                            negatives=negative_copy)

    return locals(), False

def has_converged(loglik_history, convergence_threshold, window_size=5):
    loglik_history = loglik_history[-window_size:]
    window_size = len(loglik_history)
    relative_change = np.mean(
        np.abs(np.diff(loglik_history)) / np.abs(loglik_history[:-1]))

    if relative_change <= convergence_threshold:
        logging.info(("Convergence reached; average absrel change in loglik in "
               "%i past iterations: %g" % (window_size, relative_change)))
        return True
    else:
        return False


# @profile
def _do_learning(args):
    """
    Perform learning of embeddings using stochastic gradient ascent
    """
    model = args["model"]
    alpha_train = args.get("alpha_train", .1)
    alpha_val = args.get("alpha_val", 0.)
    beta_train = args.get("beta_train", .1)
    gamma_train = args.get("gamma_train", .1)
    proc = args.get("proc", 0)
    train_data = args["train_data"]
    train_dataset = args.get("train_dataset", None)
    val_data = args.get("val_data", None)
    val_dataset = args.get("val_dataset", None)
    test_data = args.get("test_data", None)
    test_dataset = args.get("test_dataset", None)

    buckets = args.get("buckets", None)
    num_bootstraps = args.get("num_bootstraps", 20)
    disable_eval = args.get("disable_eval", False)
    inference = args.get("inference", False)
    num_iterations = args.get("num_iterations", 1000)
    learning_rate = args.get("learning_rate", 1e-3)
    momentum = args.get("momentum", 0.95)
    patience = args.get("momentum", 10)
    reduce_lr_factor = args.get("reduce_lr_factor", 0.75)
    convergence_threshold = args.get("convergence_threshold", 1e-6)
    use_early_stopping = args.get("use_early_stopping", False)
    eval_freq = args.get("eval_freq", 40)

    if inference:
        if test_data is None or buckets is None:
            raise ValueError
    if val_data is None:
        disable_eval = True

    torch.manual_seed(1234 + proc) 

    optimizer = torch.optim.Adam(model.parameters(),
                                 learning_rate, amsgrad=False)

    if val_data is not None:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', patience=patience, verbose=True,
            factor=reduce_lr_factor)

    logging.info(optimizer)

    loglik_history = []

    pid = os.getpid()
    prefix = "[proc%0i(pid=%i)] " % (proc, os.getpid())

    iteration = 0
    abort = False
    epoch = 0
    curr_auc = None
    tic = time.time()
    while not abort and iteration < num_iterations:
        logging.info("Epoch #%02i" % (epoch + 1))
        epoch += 1

        for minibatch_baskets in train_data:
            if iteration >= num_iterations:
                break

            if len(minibatch_baskets) == 0:
                break

            model.train() 
            optimizer.zero_grad()

            # 计算损失
            tic_ = time.time()
            minibatch_log_likelihood = model.compute_log_likelihood(
                model, minibatch_baskets, mapped=False,
                alpha_regularization=alpha_train,
                beta_regularization=beta_train,
                gamma_regularization=gamma_train)
            time_loss = time.time() - tic_
            tic_ = time.time()
            (-minibatch_log_likelihood).backward()
            optimizer.step()
            time_backprop = time.time() - tic_

            if val_data is not None:
                avg_val_log_likelihood = model.compute_log_likelihood(
                    model, val_data, mapped=False, alpha_regularization=0,
                    beta_regularization=0).item()

                loglik_history.append(avg_val_log_likelihood)
                logging.info("%sAvg loglik for val at iteration %s: %g | %f" % (
                    prefix, iteration, avg_val_log_likelihood, time.time()-tic) + f" | time_iter: {time_loss+time_backprop:.4f}, time_loss: {time_loss:.4f}, time_backprop: {time_backprop:.4f}")
                scheduler.step(float(avg_val_log_likelihood))

            # 验证
            if not disable_eval:
                artifacts, abort = eval_model(
                    model, val_data, eval_freq=eval_freq,
                    env=dict(prefix=prefix,
                             iteration=iteration,
                             loglik_history=loglik_history,
                             curr_auc=curr_auc),
                    inference=inference, test_data=test_data,
                    buckets=buckets, num_bootstraps=1)
                curr_auc = artifacts["curr_auc"]

            if abort:
                break

            if has_converged(loglik_history, convergence_threshold):
                abort = True
                break

            iteration += 1

    if not disable_eval:
        eval_model(model, val_data, eval_freq=eval_freq,
                   env=dict(prefix=prefix,
                            iteration=iteration),
                   inference=inference, test_data=test_data,
                   buckets=buckets, num_bootstraps=num_bootstraps)


def do_learning(model, hogwild=False, parallel_backend="mp.Pool",
                **kwargs):
    num_iterations = kwargs.pop("num_iterations", 1000)

    if hogwild:
        logging.info("Enabling HogWild parallel training")
        logging.info("Forcing torch num_threads to 1")
        num_workers =  model.num_threads
        model.num_threads = 1
        torch.set_num_threads(model.num_threads)
    else:
        num_workers = 1


    num_iterations_per_worker = ceil(float(num_iterations) / num_workers)
    kwargs["num_iterations"] = num_iterations_per_worker

    if num_workers == 1:
        results = [_do_learning({"model": model, "proc": 0, **kwargs})]
    else:
        if parallel_backend == "joblib.Parallel":
            results = Parallel(n_jobs=num_workers)(delayed(_do_learning)(
                {"model": model, "proc": proc,
                 "num_iterations": num_iterations_per_worker,
                 **kwargs}) for proc in range(num_workers))
        elif parallel_backend == "mp.Pool":
            pool = mp.Pool(processes=num_workers)
            jobs = [{"model": model, "proc": proc,
                     "num_iterations": num_iterations_per_worker,
                     **kwargs} for proc in range(num_workers)]
            results = pool.map(_do_learning, jobs)
        else:
            raise NotImplementedError(parallel_backend)

    if hogwild:
        model.num_threads = num_workers

    return model, None

