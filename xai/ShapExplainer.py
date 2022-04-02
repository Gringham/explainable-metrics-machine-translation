import os

from metrics.collection.MetricWrapper import MetricWrapper
import shap
import pandas as pd
import numpy as np

from project_root import ROOT_DIR
from xai.Explainer import Explainer
from xai.evaluation.eval4nlp_evaluate import evaluate_mlqe_auc
from xai.util.corpus_explainer import explain_corpus


class ShapExplainer(Explainer):
    '''
    SHAP explainer for metrics, using the shap library https://github.com/slundberg/shap by:
    Scott M Lundberg and Su-In Lee. “A Unified Approach to Interpreting Model Predictions”. In: Advances
    in Neural Information Processing Systems 30. Ed. by I. Guyon, U. V. Luxburg, S. Bengio, H. Wallach,
    R. Fergus, S. Vishwanathan, and R. Garnett. Curran Associates, Inc., 2017, pp. 4765–4774. url:
    http://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions.pdf.

    The functions build_feature and masker are based on the implementation in:
    https://github.com/yg211/explainable-metrics
    '''
    def __init__(self, mask_string=''):
        self.MW = MetricWrapper()
        self.mask_string = mask_string

    def build_feature(self, sent1, sent2=None):
        '''
        Creates a pandas dataframe where each token of sent1 and optionally sent2 is written to a different column
        :param sent1: The tokens of the first sentence are prepended by s1_
        :param sent2: The tokens of the second sentence are prepended by s2_ . The second sentence is not used in the
                      final implementation
        :return:
        '''
        tdict = {}

        sent1_tokens = sent1.split(' ')
        self.l1len = len(sent1_tokens)

        for i in range(len(sent1_tokens)):
            tdict['s1_{}'.format(i)] = sent1_tokens[i]

        if sent2:
            sent2_tokens = sent2.split(' ')
            for i in range(len(sent2_tokens)):
                tdict['s2_{}'.format(i)] = sent2_tokens[i]
        return pd.DataFrame(tdict, index=[0])

    def masker(self, mask, x, sent2=False):
        '''
        replaces the tokens in x with a mask string, where indicated by a mask
        :param mask: A mask that indicates replacement positions
        :param x: Tokens of sentence 1
        :param sent2: If specified, it is assumed that x contains the tokens of sentence 2 as well. And they are processed
                      by putting [SEP] inbetween. This is not used in the final implementation
        :return:
        '''
        tokens = []
        for mm, tt in zip(mask, x):
            if mm:
                tokens.append(tt)
            else:
                tokens.append(self.mask_string)
        if sent2 == False:
            sentence_pair = ' '.join(tokens)
        else:
            s1 = ' '.join(tokens[self.l1len:])
            s2 = ' '.join(tokens[:self.l1len])
            sentence_pair = s1 + '[SEP]' + s2
        return pd.DataFrame([sentence_pair])

    def determine_method_and_features(self, hyp, max_exact=10):
        '''
        Precomputes sentences and methods in input format
        :param hyp: A hypothesis
        :param max_exact: The number of maximum features for which exact shap should be used
        :return: Dictionary with the method and pandas dataframes for each input sentence
        '''
        pre_dict = {'method': 'auto', 'hyp_features':self.build_feature(hyp)}
        if len(hyp.split()) <= max_exact:
            pre_dict['method'] = 'exact'
        return pre_dict

    def explain_hyp(self, hyp, metric, pre_dict={'method': 'exact'}):
        '''
        Need to unwrap np array during metric calculation --> to list and afterwards wrap it back
        as features are precomputed, the actual hyp is gotten from pre_dict
        :param hyp: dummy value for hypothesis (real hypothesis comes from pre_dict)
        :param metric: The metric to explain
        :param pre_dict: Precomputed hypotheses and method
        :return: explanation for the current sample
        '''
        fixed_src_metric = lambda x: np.array(metric([a[0] for a in x.tolist()]))
        explainer = shap.Explainer(fixed_src_metric, self.masker, algorithm=pre_dict['method'])
        return explainer(pre_dict['hyp_features'])

    def apply_explanation(self, df, metrics=None, max_exact=10, recover=False):
        '''
        :param df: Dataframe with samples to explain
        :param metrics: The metrics to explain. If none, all metrics known to MetricWrapper will be considered
        :param max_exact: The maximum number of features to which exact shap is computed
        :param recover: Whether to recover existing computations
        :return: Dictionary with feature importance explanations
        '''
        explanations = self.MW.apply_hyp_explainer(df,
                                                   self.explain_hyp,
                                                   metrics=metrics,
                                                   precompute=lambda x: self.determine_method_and_features(x, max_exact),
                                                   recover=recover)


        # compute the original scores separately in case of shap (as it doesn't seem to keep them internally)
        scores = self.MW.evaluate(df, metrics)
        if 'XMOVERSCORE' in metrics:
            scores['XMOVERSCORE'] = scores['xmoverscore_clp2_lm']

        # Unpacking the explanation object
        attributions = []
        for x in range(len(explanations)):
            print('Sample:', explanations[x])
            attribution = {'src': explanations[x]['src'],
                           'ref': explanations[x]['ref'],
                           'hyp': explanations[x]['hyp'],
                           'metrics': {key: {'attributions': list(zip(value[0].values, value[0].data)),
                                             'predicted_score': float(sum(value[0].values) + value[0].base_values),
                                             'base':float(value[0].base_values),
                                             'score': scores[key][x]
                                             }
                                       for key, value in explanations[x]['metrics'].items()},
                           'times': {key: {'time': value}
                                       for key, value in explanations[x]['times'].items()}
                           }

            print(attribution)
            attributions.append(attribution)
        return attributions



if __name__ == '__main__':
    SE = ShapExplainer(mask_string='')

    paths = [#(os.path.join(ROOT_DIR, 'metrics/corpora/pandas_corpora/eval4NLP/eval4nlp_test_et-en.tsv'), 1000, 'et-en')
     #(os.path.join(ROOT_DIR, 'metrics/corpora/pandas_corpora/eval4NLP/eval4nlp_test_ro-en.tsv'), 1000, 'ro-en'),
     #(os.path.join(ROOT_DIR, 'metrics/corpora/pandas_corpora/eval4NLP/eval4nlp_test_ru-de.tsv'), 1180, 'ru-de'),]
     (os.path.join(ROOT_DIR, 'metrics/corpora/pandas_corpora/eval4NLP/eval4nlp_test_de-zh.tsv'), 1410, 'de-zh')]

    for path, n, lp in paths:
        explain_corpus(SE,
                       recover=False,
                       metrics=['TRANSLATIONMETEOR'],
                       from_row=0, to_row=n,
                       outfile=lp + '_test_attributions_shap_wbw.json',
                       mlqe_pandas_path=path, max_exact=10)

        evaluate_mlqe_auc([
            os.path.join(ROOT_DIR,'xai/output/explanations/0_' + str(
                n - 1) + '_' + lp + '_test_attributions_shap_wbw.json')],
            start=0, end=n, invert=True, f=100,
            corpus_path=path[:-4]+'-gold.tsv')
