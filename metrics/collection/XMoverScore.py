from os import path

import torch
from mosestokenizer import MosesDetokenizer
import truecase

from metrics.collection.MetricClass import MetricClass
import numpy as np

from metrics.corpora.wmt.download_lang_from_sacrebleu import lp

import os
from project_root import ROOT_DIR


class XMoverScore(MetricClass):
    '''A wrapper for XMoverScore (https://github.com/AIPHES/ACL20-Reference-Free-MT-Evaluation), by:
    Wei Zhao, Goran Glavaš, Maxime Peyrard, Yang Gao, Robert West, and Steffen Eger. “On the Lim-
    itations of Cross-lingual Encoders as Exposed by Reference-Free Machine Translation Evaluation”.
    In: Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics. Online:
    Association for Computational Linguistics, July 2020, pp. 1656–1671.
    url: https://www.aclweb.org/anthology/2020.acl-main.151'''
    ref_based = False
    name = 'XMOVERSCORE'

    def __init__(self, bs=16, layer=12, xlm=False, drop_punctuation=True, model_name=None, k=2, extension='.map'):
        '''
        :param bs: batch size
        :param layer: layer, only use this parameter with the xlm version
        :param xlm: use the xlmr version of xms
        :param drop_punctuation: drop punctuation and subwords (standard). Has no effect in the xlm version. The xlm version keeps them per default
        :param model_name: The model name for the xlmr model, when xlm mode is used
        :param k: The number of sentences used to train the remapping (eg 2k ---> 2000 sentences where used)
        :param extension:
        '''
        self.bs = bs
        self.k = k
        if not xlm:
            from metrics.collection.metrics_libs.xmoverscore.scorer import XMOVERScorer
            self.scorer = XMOVERScorer('bert-base-multilingual-cased', 'gpt2', False, drop_punctuation=drop_punctuation)
        else:
            from metrics.collection.metrics_libs.xmoverscore_xlmr.scorer import XMOVERScorer
            self.scorer = XMOVERScorer(model_name, 'gpt2', False, drop_punctuation=drop_punctuation)

        self.layer = layer
        self.xlm = xlm
        self.extension=extension

    def __call__(self, src, hyp, lp='de-en',
                 mapping_path=os.path.join(ROOT_DIR,'metrics/collection/metrics_libs/xmoverscore/mapping'),
                 mode=['ALL'], preprocess=True):
        '''
        :param ref: A list of strings with reference sentences
        :param hyp: A list of strings with hypothesis sentences
        :return: A list of list of XMS Scores [CLP_unigram, CLP_bigram, UMD_unigram, UMD_bigram, CLP_unigram_lm,...]
        '''

        if not self.xlm:
            # projection, bias = self.load_mapping(lp, os.path.join(ROOT_DIR,'metrics/collection/metrics_libs/xmoverscore_xlmr/mapping',k=self.k)
            projection, bias = self.load_mapping(lp, mapping_path, k=self.k, extension=self.extension)

        else:
            projection, bias = self.load_mapping(lp,
                                                 os.path.join(ROOT_DIR,'metrics/collection/metrics_libs/xmoverscore_xlmr/mapping'),
                                                 k=self.k, extension=self.extension)

        if preprocess:
            src_preprocessed, hyp_preprocessed = self.preprocess(lp, src, hyp)
        else:
            src_preprocessed = src
            hyp_preprocessed = hyp
        lm_perplexity = self.scorer.compute_perplexity(hyp_preprocessed, bs=1)

        if 'ALL' in mode:
            mode = ['CLP_1', 'CLP_2', 'UMD_1', 'UMD_2']
        results = []

        if 'CLP_1' in mode:
            results.append(self.scorer.compute_xmoverscore('CLP', projection, bias, src_preprocessed, hyp_preprocessed,
                                                           ngram=1, bs=self.bs, layer=self.layer))
            results.append(self.metric_combination(results[-1][0], lm_perplexity, [1, 0.1]).tolist())
        else:
            results += [[], []]

        if 'CLP_2' in mode:
            results.append(self.scorer.compute_xmoverscore('CLP', projection, bias, src_preprocessed, hyp_preprocessed,
                                                           ngram=2, bs=self.bs, layer=self.layer))
            results.append(self.metric_combination(results[-1][0], lm_perplexity, [1, 0.1]).tolist())
        else:
            results += [[], []]

        if 'UMD_1' in mode:
            results.append(self.scorer.compute_xmoverscore('UMD', projection, bias, src_preprocessed, hyp_preprocessed,
                                                           ngram=1, bs=self.bs, layer=self.layer))
            results.append(self.metric_combination(results[-1][0], lm_perplexity, [1, 0.1]).tolist())
        else:
            results += [[], []]

        if 'UMD_2' in mode:
            results.append(self.scorer.compute_xmoverscore('UMD', projection, bias, src_preprocessed, hyp_preprocessed,
                                                           ngram=2, bs=self.bs, layer=self.layer))
            results.append(self.metric_combination(results[-1][0], lm_perplexity, [1, 0.1]).tolist())
        else:
            results += [[], []]

        return results

    def preprocess(self, lp, src, hyp):
        s, t = lp.split('-')
        with MosesDetokenizer(s) as detokenize:
            src_detok = [detokenize(sent.split(' ')) for sent in src]
        with MosesDetokenizer(t) as detokenize:
            hyp_detok = [detokenize(sent.split(' ')) for sent in hyp]

        hyp_detok = [truecase.get_true_case(sent) for sent in hyp_detok]
        return src_detok, hyp_detok

    def load_mapping(self, lp, mapping_path, k='2', extension='.map'):
        s, t = lp.split('-')
        try:
            temp = np.loadtxt(path.join(mapping_path, 'europarl-v7.' + s + '-' + t + '.' + str(k) + 'k.' + str(
                self.layer) + '.BAM' + extension))
        except:
            temp = np.load(path.join(mapping_path, 'europarl-v7.' + s + '-' + t + '.' + str(k) + 'k.' + str(
                self.layer) + '.BAM' + extension))
        projection = torch.tensor(temp, dtype=torch.float).to('cuda:0')

        try:
            temp = np.loadtxt(
                path.join(mapping_path, 'europarl-v7.' + s + '-' + t + '.' + str(k) + 'k.' + str(
                    self.layer) + '.GBDD' + extension))
        except:
            temp = np.load(path.join(mapping_path,
                                     'europarl-v7.' + s + '-' + t + '.' + str(k) + 'k.' + str(
                                         self.layer) + '.GBDD' + extension))
        bias = torch.tensor(temp, dtype=torch.float).to('cuda:0')
        return projection, bias

    def metric_combination(self, a, b, alpha):
        return alpha[0] * np.array(a) + alpha[1] * np.array(b)

    def get_abstraction(self, src, lp='de-en',
                        mapping_path=os.path.join(ROOT_DIR,'metrics/collection/metrics_libs/xmoverscore/mapping')):
        '''
        As this function needs a language pair for xmoverscore we are overwriting the base
        :param src: A source to be used with every value
        :param ref: A ref to be used with every value
        :return: A function only depending on a list of references. Returns clp2 scores
        '''

        return lambda hyp: self.__call__([src] * len(hyp), hyp, lp=lp, mapping_path=mapping_path, mode='CLP_2')[3]


if __name__ == '__main__':
    b = XMoverScore()

    # Sample using ref and hyp lists (Returns: [CLP1(sent score, hyp_scores, src_scores), CLP1+LM(Sent Score), CLP2(sent score,
    # hyp_scores, src_scores (might be buggy for 2-grams)) , CLP2+LM(Sent scores), UMD1(sent score, hyp_scores, src_scores), UMD1+LM(Sent Score), UMD2(sent score,
    # hyp_scores, src_scores (might be buggy for 2-grams)), UMD2+LM(Sent scores)
    print(b(["A test sentence"], ["A simple sentence for test"]))
    #[([0.12644830844820176],
    #  [[('A', 0.72940993309021), ('simple', 0.924140214920044), ('sentence', 0.7055093050003052),
    #                           ('for', 1.0305591821670532), ('test', 0.8466565608978271)]],
    #  [[('A', 0.72940993309021), ('test', 0.8466565608978271), ('sentence', 0.7055093050003052)]]),
    # [-0.602806175728434],
    # ([0.1616856754114816], [
    #    [('A', 0.7950088977813721), ('simple', 0.7707442045211792), ('sentence', 0.7707442045211792),
    #     ('for', 0.8122541904449463), ('test', 0.8391879796981812)]],
    #  [[]]),
    # [-0.5675688087651541], (
    # [0.47764761482186513], [
    #     [('A', 0.3220022916793823), ('simple', 0.6680176258087158), ('sentence', 0.358559250831604),
    #      ('for', 0.7766213417053223), ('test', 0.5934580564498901)]],
    # [[('A', 0.3220022916793823), ('test', 0.5934580564498901), ('sentence', 0.358559250831604)]]),
    # [-0.2516068693547706],
    # ([0.5090889277570457], [
    #    [('A', 0.3149862289428711), ('simple', 0.42935431003570557), ('sentence', 0.42935431003570557),
    #     ('for', 0.4729980230331421), ('test', 0.5588963031768799)]], [[]]),
    #     [-0.22016555641959001]]

    # Sample using a fixed reference for a list of hypotheses
    b_trimmed = b.get_abstraction("Ein Test Satz", 'de-en')
    print(b_trimmed(["A simple sentence for test", "Another simple sentence for test", 'A test sentence']))
    # Only clp2 sentence scores are considered for the abstracted scores
    # [-0.6367431846654992, -0.7090483289647004, -0.6119382109546662]
