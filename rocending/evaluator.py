# encoding=utf8

from src.utils.misc import parallel_run
from .answer_util import max_em
from datasets import load_metric
from rouge import Rouge

class EvaluateTool(object):
    def __init__(self):
        pass

    def evaluate(self, preds, golds, score=False):
        em = compute_em(preds, golds, score=False)
        em_max = compute_max_em(preds, golds, score=False)
        blue_4 = compute_bleu(preds, golds, score=False)
        rouge_l = compute_rouge(preds, golds, score=False)
        return em, em_max, blue_4, rouge_l


def compute_em(preds, golds, score=False):
        golds = [gold["target"] for gold in golds]
        em_results = parallel_run(eval_em_single, list(zip(preds, golds)))
        if score is True:
            return em_results
        else:
            return {"em": sum(em_results) / len(golds)}

def compute_max_em(preds, golds, score=False):
        golds = [gold["target"] for gold in golds]
        eval_results = parallel_run(eval_single, list(zip(preds, golds)))
        metrics = {"max_em": sum(eval_results) / len(golds)}
        return metrics

def compute_bleu(preds, golds, score=False):
        # character-level 3-bleu
        bleu = load_metric('src/metrics/bleu.py')
        if score is True:
            bleu_score = []
            for i in range(len(golds)):
                predictions = [[ch for ch in preds[i]]]
                references = [[[ch for ch in golds[i]['target']]]]
                bleu_score.append(bleu.compute(predictions=predictions, references=references)['bleu'])
            return bleu_score
        else:
            predictions = [[ch for ch in text] for text in preds]
            references = [[[ch for ch in entry['target']]] for entry in golds]
            return bleu.compute(predictions=predictions, references=references, max_order=1)

def compute_rouge(preds, golds, score=False):
        r = Rouge(["rouge-l"])
        predictions = preds
        references = [entry['target'] for entry in golds]
        return r.get_scores(predictions, references)
        
def eval_em_single(args):
    pred, gold = args
    return pred.strip() == gold

def eval_single(args):
    pred, gold_answers = args
    return max_em(pred, gold_answers)