import argparse
import os

import dill
import numpy as np
import statistics

from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import pearsonr

from metrics.collection.MetricWrapper import MetricWrapper
from project_root import ROOT_DIR
from xai.evaluation.submissions.prepare_submission import extract_attributions
from xai.util.json_reader import load_files
import pandas as pd
import tqdm

#from xai.xai_libs.custom_lime import Perturber

# The first functions declared here are taken from https://github.com/eval4nlp/SharedTask2021/blob/main/scripts/evaluate.py
# The end of this block is marked by ############################

def read_sentence_data(gold_sent_fh, model_sent_fh):
    gold_scores = [float(line.strip()) for line in gold_sent_fh]
    model_scores = [float(line.strip()) for line in model_sent_fh]
    assert len(gold_scores) == len(model_scores)
    return gold_scores, model_scores


def read_word_data(gold_explanations_fh, model_explanations_fh):
    gold_explanations = [list(map(int, line.split())) for line in gold_explanations_fh]
    model_explanations = [list(map(float, line.split())) for line in model_explanations_fh]
    assert len(gold_explanations) == len(model_explanations)
    for i in range(len(gold_explanations)):
        assert len(gold_explanations[i]) == len(model_explanations[i])
        assert len(gold_explanations[i]) > 0
    return gold_explanations, model_explanations


def validate_word_level_data(gold_explanations, model_explanations):
    valid_gold, valid_model = [], []
    for gold_expl, model_expl in zip(gold_explanations, model_explanations):
        #print(len(gold_expl), len(model_expl), gold_expl)
        if sum(gold_expl) == 0 or sum(gold_expl) == len(gold_expl) or len(gold_expl)!=len(model_expl):
            continue
        else:
            valid_gold.append(gold_expl)
            valid_model.append(model_expl)
    return valid_gold, valid_model


def compute_auc_score(gold_explanations, model_explanations):
    res = 0
    for i in range(len(gold_explanations)):
        res += roc_auc_score(gold_explanations[i], model_explanations[i])
    return res / len(gold_explanations)


def compute_ap_score(gold_explanations, model_explanations):
    res = 0
    for i in range(len(gold_explanations)):
        res += average_precision_score(gold_explanations[i], model_explanations[i])
    return res / len(gold_explanations)


def compute_rec_topk(gold_explanations, model_explanations):
    res = 0
    for i in range(len(gold_explanations)):
        idxs = np.argsort(model_explanations[i])[::-1][:sum(gold_explanations[i])]
        res += len([idx for idx in idxs if gold_explanations[i][idx] == 1]) / sum(gold_explanations[i])
    return res / len(gold_explanations)


def evaluate_word_level(gold_explanations, model_explanations):
    gold_explanations, model_explanations = validate_word_level_data(gold_explanations, model_explanations)
    auc_score = compute_auc_score(gold_explanations, model_explanations)
    ap_score = compute_ap_score(gold_explanations, model_explanations)
    rec_topk = compute_rec_topk(gold_explanations, model_explanations)
    # print('AUC score: {:.3f}'.format(auc_score))
    # print('AP score: {:.3f}'.format(ap_score))
    # print('Recall at top-K: {:.3f}'.format(rec_topk))
    return auc_score, ap_score, rec_topk


def evaluate_sentence_level(gold_scores, model_scores):
    corr = pearsonr(gold_scores, model_scores)[0]
    print('Pearson correlation: {:.3f}'.format(corr))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gold_explanations_fname', type=argparse.FileType('r'), required=True)
    parser.add_argument('--model_explanations_fname', type=argparse.FileType('r'), required=True)
    parser.add_argument('--gold_sentence_scores_fname', type=argparse.FileType('r'), required=True)
    parser.add_argument('--model_sentence_scores_fname', type=argparse.FileType('r'), required=True)
    args = parser.parse_args()
    gold_explanations, model_explanations = read_word_data(args.gold_explanations_fname, args.model_explanations_fname)
    gold_scores, model_scores = read_sentence_data(args.gold_sentence_scores_fname, args.model_sentence_scores_fname)
    evaluate_word_level(gold_explanations, model_explanations)
    evaluate_sentence_level(gold_scores, model_scores)


#######################################################################################################################


def evaluate_mlqe_auc(result_paths,
                       start=0,
                       end=500,
                       invert=True,
                       f=0,
                       corpus_path=os.path.join(ROOT_DIR,'metrics/corpora/pandas_corpora/mlqe/mlqe_et_zeros_dropped.tsv')):
    '''
    :param result_paths: where to load explanations from
    :param start: The start row to consider in the ground truth
    :param end: The end row to consider in the ground truth
    :param invert: Whether token level scores should be inverted
    :param f: Just a counter variable that influences the output file name
    :param corpus_path: The path to a pandas corpus with ground truth (see metrics/corpora/pandas_corpora)
    :return:
    '''


    # Load groundtruth that was prepared as pandas df
    gt = pd.read_csv(corpus_path, delimiter='\t')[start:end]
    explanations = load_files(result_paths)

    # Compute the pearson corellation with DA labels
    pearsonscores = {}
    if 'score' in explanations[0]['metrics'][list(explanations[0]['metrics'].keys())[0]]:
        scores_per_key = {key: [exp['metrics'][key]['score'] for exp in explanations] for key in
                          explanations[0]['metrics'].keys()}
        for key in scores_per_key.keys():
            pearsonscores[key] = pearsonr(scores_per_key[key], gt['DA'])[0]


    # Preprocess and align the token-level scores with the words of the hypothesis
    src_attributions_per_key = None
    attributions_per_key = extract_attributions(gt, explanations, invert=invert)
    if 'src_attributions' in explanations[0]['metrics'][list(explanations[0]['metrics'].keys())[0]]:
        src_attributions_per_key = extract_attributions(gt, explanations, attribution_key='src_attributions', comp='SRC', invert=invert)

    res = []
    for key in attributions_per_key.keys():
        try: # the tags may be saved in different ways in the pd dataframe
            auc, ap, rec_tok = evaluate_word_level(
                [[int(s) for s in s[1:-1].split(', ')] for s in gt['TAGS_HYP']], attributions_per_key[key])
        except:
            auc, ap, rec_tok = evaluate_word_level(
                [[int(s) for s in s.split()] for s in gt['TAGS_HYP']], attributions_per_key[key])

        if src_attributions_per_key and 'TAGS_SRC' in gt:
            try:
                auc_src, ap_src, rec_tok_src = evaluate_word_level(
                    [[int(s) for s in s[1:-1].split(', ')] for s in gt['TAGS_SRC']], src_attributions_per_key[key])
            except:
                # for the test data the split is called a bit differently, as we hadn't had it transformed to list before
                auc_src, ap_src, rec_tok_src = evaluate_word_level(
                    [[int(s) for s in s.split()] for s in gt['TAGS_SRC']], src_attributions_per_key[key])
            if key in pearsonscores:
                res.append([key, auc, ap, rec_tok, auc_src, ap_src, rec_tok_src, pearsonscores[key]])
            else:
                res.append([key, auc, ap, rec_tok, auc_src, ap_src, rec_tok_src, np.nan])
        else:
            if key in pearsonscores:
                res.append([key, auc, ap, rec_tok, pearsonscores[key]])
            else:
                res.append([key, auc, ap, rec_tok, np.nan])

    if src_attributions_per_key and 'TAGS_SRC' in gt:
        df = pd.DataFrame(res, columns=['Metric', 'AUC', 'AP', 'REC_TOPK','AUC_SRC', 'AP_SRC', 'REC_TOPK_SRC', 'PEARSON']).round(3)
    else:
        df = pd.DataFrame(res, columns=['Metric', 'AUC', 'AP', 'REC_TOPK', 'PEARSON']).round(3)

    print(df)
    df.to_csv(
        os.path.join(ROOT_DIR,'xai/output/eval_auc/auc_et_non_inverse' + str(
            f) + "_" + str(start) + "_" + str(end) + ".tsv"), sep='\t', index=False)

def evaluate_mlqe_time(result_paths):
    '''
    :param result_paths: path to load explanations from
    :return: time per explainability technique
    '''
    explanations = load_files(result_paths)

    times_per_key = {
        key: sum([explanations[x]['times'][key]['time'] for x in
              range(len(explanations))])/len(explanations) for key in explanations[0]['metrics']}

    for key, metric in times_per_key.items():
        print(key, metric)

def eval_multi(paths, start=0, end=100, invert=True):
    '''
    :param paths: a list of list of paths with explanations for a given part of the MLQE dataset as loaded by the MLQE loader
    :return: auc score files for all specified files
    '''
    for x in tqdm.tqdm(range(len(paths))):
        print('-------------------------')
        print(paths[x])
        evaluate_mlqe_auc(paths[x], start, end, f=x, invert=invert)


if __name__ == '__main__':
    eval_multi([os.path.join(ROOT_DIR,'xai/output/explanations/0_99_mlqe_et_attributions_shap_2.json')])