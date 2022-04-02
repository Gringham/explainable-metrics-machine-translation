from metrics.collection.MetricWrapper import MetricWrapper
from metrics.evaluation.ProduceBins import get_bins_from_corpus
from xai.Explainer import Explainer
from xai.util.corpus_explainer import explain_corpus

from textattack.models.wrappers import ModelWrapper
import numpy as np
from scipy.stats import norm
from textattack import Attack, AttackArgs
from textattack.search_methods import GreedySearch
from textattack.goal_functions import UntargetedClassification, TargetedClassification
from textattack.transformations import WordSwapEmbedding
from xai.xai_libs.TextAttack.bert_attack_li_2020 import BERTAttackLi2020

import os
from project_root import ROOT_DIR


class MetricModelWrapper(ModelWrapper):
    '''
    A custom textattack modelwrapper, that discretizes a metric, i.e. turns regression into a classification problem.
    An alternative implementation would be to implement a custom goal for regression (e.g. change the score as much as possible,
    or in a specific range)
    '''
    def __init__(self, metric):
        # These are overwritten by setting them directly in the explain function below
        self.metric = metric # attacked metric
        self.model = None
        self.current_ref = None
        self.binning_mode = 'corpus'
        self.fixed_lower = 0 # lower boundary for fixed binning mode
        self.fixed_upper = 1 # upper boundary for fixed binning mode
        self.loose_span = 1 # Span size for loose binning mode
        self.bins = 5 # Number of bins
        self.std = 0.1 # Standard deviation for fixed and loose binning mode
        self.orig = None # The original score for the attacked sample
        self.corpus_bins = None # A list of class boundaries when using corpus binning mode

    def __call__(self, text_input_list, batch_size=None):
        '''
        :param text_input_list:  A list of input texts
        :param batch_size:  The batch size
        :return:  a discretized score. In case no binning mode is defined we return the original score and predictions for
                  all modes
        '''
        hyps = text_input_list
        score = self.metric(hyps)
        if self.binning_mode == 'fixed':
            return self.fixed_bins(score)
        if self.binning_mode == 'loose':
            return self.loose_bins(score, self.orig)
        if self.binning_mode == 'corpus':
            return self.corpus_bins_mode(score, self.corpus_bins)

        # If this class is not used in prediction, I return the argmax of the defaults
        if self.corpus_bins :
            return score, [int(n) for n in np.argmax(self.fixed_bins(score), axis=1)], \
                   [int(n) for n in np.argmax(self.corpus_bins_mode(score, self.corpus_bins), axis=1)], \
                   [int(np.argmax(self.loose_bins([s], s))) for s in score]
        else:
            return score, [int(n) for n in np.argmax(self.fixed_bins(score), axis=1)], \
                   [], \
                   [int(np.argmax(self.loose_bins([s], s))) for s in score]

    def fixed_bins(self, scores):
        '''
        provides "probabilities" for the result to lie within fixed intervals between a specified boundary
        smaller probabilities indicate that the score is more likely to jump
        the probability for each interval is modeled with a normal distribution centered around the middle of the score
        each interval gets the cumulative probability
        :param scores: a list of scores to discretize
        :return: discretized scores
        '''
        bins = np.linspace(self.fixed_lower, self.fixed_upper, self.bins + 1)
        binned_stats = [
            np.array([norm(score, self.std).cdf(bins[x + 1]) - norm(score, self.std).cdf(bins[x]) for x in
                      range(len(bins) - 1)])
            for score in scores]

        # scaling with offsets to sum to 1
        # https://stackoverflow.com/questions/46160717/two-methods-to-normalise-array-to-sum-total-to-1-0
        return [((b - b.min()) / (b - b.min()).sum()).tolist() for b in binned_stats]

    def loose_bins(self, scores, orig):
        '''
        provides "probabilities" for the result to lie within fixed intervals around a loose central score
        smaller probabilities indicate that the score is more likely to jump
        the probability for each interval is modeled with a normal distribution centered around the middle of the score
        each interval gets the cumulative probability
        :param scores: a list of scores to discretize
        :return: discretized scores
        '''
        bins = np.linspace(orig - (self.loose_span / 2), orig + (self.loose_span / 2), self.bins + 1)
        binned_stats = [
            np.array([norm(score, self.std).cdf(bins[x + 1]) - norm(score, self.std).cdf(bins[x]) for x in
                      range(len(bins) - 1)])
            for score in scores]

        # scaling with offsets to sum to 1
        # https://stackoverflow.com/questions/46160717/two-methods-to-normalise-array-to-sum-total-to-1-0
        return [((b - b.min()) / (b - b.min()).sum()).tolist() for b in binned_stats]

    def corpus_bins_mode(self, scores, corpus_bins):
        '''
        provides "probabilities" for the result to lie within intervals that contained a equal number of metric scores on
        a specific corpus. When using it, it is assumed, that the attacked sample was part of this corpus.
        :param scores: a list of scores to discretize
        :param corpus_bins: a list of interval objects
        :return: discretized scores
        '''
        intervals = []
        for score in scores:
            res = []
            side = None
            for i in range(len(corpus_bins)):
                # Check if the score is part of the current interval
                if score in corpus_bins[i]:
                    # get the percentage of the interval where the score lied (e.g. on [0,1], 0.7 lies at 0.7)
                    d = corpus_bins[i].length
                    s = score - corpus_bins[i].left
                    p_full = s / d

                    # I use equally spaced "class probabilities", i.e. assign the predicted class with at least 0.5 and then
                    # assign the rest probability to the nearest neighboring class

                    # If the score is on the left side of the interval, I determine its "class probability" as 0.5 + the
                    # percentage on the interval
                    if p_full < 0.5:
                        side = 'left'
                        p = p_full + 0.5

                        # if the predicted class is 0, there is no left class so I assign a class probability of 1
                        if i == 0:
                            p = 1.0

                    # if the score is on the right side
                    else:
                        side = 'right'
                        p = (1 - p_full) + 0.5

                        # if the predicted class is the highest one, there is no right class, so I assign 1
                        if i == len(corpus_bins) - 1:
                            p = 1.0

                    res.append(p)
                else:
                    res.append(0)

            if side == 'left':
                # assign rest 'probability' to left class
                for i in range(1, len(res)):
                    if res[i] > 0:
                        res[i - 1] = 1 - res[i]

            elif side == 'right':
                # assign rest 'probability' to right class
                for i in reversed(range(0, len(res) - 1)):
                    if res[i] > 0:
                        res[i + 1] = 1 - res[i]

            #if score <= corpus_bins[0].mid:
            #    res[0] = 1.0
            #elif score >= corpus_bins[-1].mid:
            #    res[-1] = 1.0

            intervals.append(res)

        # print(intervals)
        return intervals


class AdversarialExplainer(Explainer):
    '''
    Attacks metrics with TextAttack (https://github.com/QData/TextAttack) by
    John Morris, Eli Lifland, Jin Yong Yoo, Jake Grigsby, Di Jin, and Yanjun Qi. “TextAttack: A Framework
    for Adversarial Attacks, Data Augmentation, and Adversarial Training in NLP”. In: Proceedings of
    the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations.
    Online: Association for Computational Linguistics, Oct. 2020, pp. 119–126. doi: 10.18653/v1/
    2020.emnlp-demos.16. url: https://aclanthology.org/2020.emnlp-demos.16.

    Explains by generating adversarial samples of different kinds of attacks for each sample in df
    '''

    def __init__(self, binning_mode='corpus', bins=3, fixed_lower=0, fixed_upper=1, loose_span=1, std=0.1, attack_types=['bert-attack'],
                 target = 0, corpus = os.path.join(ROOT_DIR,'metrics/outputs/wmt/wmt20_full_scores_raw_zh_en')):
        '''
        :param binning_mode: 'fixed', 'loose', 'corpus' or None
        :param bins: the number of bins for classification. Should be uneven in loose mode
        :param fixed_lower: the lower boundary for fixed mode
        :param fixed_upper: the upper boundary for fixed mode
        :param loose_span: the span around the original score in loose mode
        :param std:  the standard deviation for generated class probabilities
        :param attack_types: attack names in a list. It is more efficient to run them one by one, i.e. pass only one each time
        :param target: Target class
        :param corpus: The corpus to use to infer bins
        '''
        self.MW = MetricWrapper()
        self.binning_mode = binning_mode
        self.bins = bins
        self.fixed_lower = fixed_lower
        self.fixed_upper = fixed_upper
        self.loose_span = loose_span
        self.std = std
        self.attack_types = attack_types
        self.target = target
        self.corpus_bins = get_bins_from_corpus(n=self.bins, corpus=corpus)
        self.metric_name = None
        self.MMW = None
        self.attack = None

    def apply_attack(self, hyp, orig_score, pred_class, model_wrapper, recipe='bert-attack', target=None,
                     attack_defined=False):
        '''
        :param hyp: hypothesis that is attacked
        :param orig_score: the original score
        :param pred_class: the original class
        :param model_wrapper: the model wrapper object
        :param recipe: the name of the attack recipe
        :param target: the target class
        :param attack_defined: if the attack is already defined, I can pass this parameter so it doesn't have to be rebuilt
        :return: Attack results
        '''


        if attack_defined:
            attack = self.attack
        else:
            model_wrapper.orig = orig_score

            if recipe == 'bert-attack':
                attack = BERTAttackLi2020.build(model_wrapper)

            elif recipe == 'textfooler':
                from textattack.attack_recipes import TextFoolerJin2019
                attack = TextFoolerJin2019.build(model_wrapper)

            elif recipe == 'textfooler_adjusted':
                from xai.xai_libs.textattack_recipes.TextFoolerAdjusted import TextFoolerJin2019Adjusted
                attack = TextFoolerJin2019Adjusted(model_wrapper)

            else:
                goal_function = UntargetedClassification(model_wrapper)
                transformation = WordSwapEmbedding()
                search_method = GreedySearch()

                attack = Attack(goal_function, [], transformation, search_method)

            if target != None:
                # repackage attack to use targetedClassification as goal
                goal_function = TargetedClassification(model_wrapper, target_class=target)
                attack = Attack(goal_function, attack.pre_transformation_constraints + attack.constraints,
                                attack.transformation, attack.search_method)

            self.attack = attack
        print(pred_class, target)
        try:
            res = attack.attack(hyp, pred_class)
        except Exception as e:
            print('Error encountered, writing None: ', e)
            res = None
        return res

    def explain(self, hyp, metric, pre_dict={}):
        '''
        # This function is passed to the apply loop. It attacks a metric for a given hypothesis
        :param hyp: hypothesis sentence
        :param metric: metric (with fixed src or ref)
        :param pre_dict: precomputed values. This is a bit hacky. As the application loop for the metrics is a nested lambda
        function, it is difficult to pass the name of the metric. So instead I pass the metric with the precomputation dict
        even though it is not really a precomputation but rather changed every loop
        :return: attack results and the original score
        '''

        if self.metric_name != pre_dict['key']:
            # if the metric name changed the model wrapper needs to be updated
            self.MMW = MetricModelWrapper(metric)
            attack_defined = False
        else:
            # otherwise, the metric needs to be updated (as the src or ref changed), but the attack can be kept loaded
            attack_defined = len(self.attack_types) == 1
            self.MMW.metric = metric

        # update the other properties
        self.MMW.binning_mode = None
        self.MMW.fixed_lower = self.fixed_lower
        self.MMW.fixed_upper = self.fixed_upper
        self.MMW.loose_span = self.loose_span
        self.MMW.bins = self.bins
        self.MMW.std = self.std
        self.metric_name = pre_dict['key']
        metric_name = self.metric_name

        if self.metric_name == 'XMOVERSCORE':
            metric_name = 'xmoverscore_clp2_lm'

        if self.binning_mode == 'corpus':
            self.MMW.corpus_bins = self.corpus_bins[0][metric_name]
        scores, fixed_classes, corpus_classes, loose_classes = self.MMW([hyp])

        attributions = {}

        # Create a dataset based on the binning mode
        if self.binning_mode == 'fixed':
            classes = fixed_classes
            self.MMW.binning_mode = 'fixed'
        elif self.binning_mode == 'corpus':
            classes = corpus_classes
            self.MMW.binning_mode = 'corpus'
        else:
            classes = loose_classes
            self.MMW.binning_mode = 'loose'

        for attack_type in self.attack_types:
            print("Running attack: ", attack_type)
            attributions[attack_type] = self.apply_attack(hyp, scores[0], classes[0], self.MMW, recipe=attack_type,
                                                          target=self.target, attack_defined=attack_defined)

        return attributions, scores[0]

    def apply_explanation(self, df, metrics=None, recover=False,
                          idf_path=os.path.join(ROOT_DIR,'metrics/collection/metrics_libs/moverscore/idfs/wmt_2020_zh_en_msidf.dill'),
                          explanation_path=os.path.join(ROOT_DIR,'xai/output/explanation_checkpoints/adv_attack_zh_en_2_to_0_ba.dill')):
        '''
        Applies adversarial attacks on metrics and
        :param df: The dataframe to attack samples on
        :param metrics: A list of metric names to explain. When none, all metrics known by MetricWrapper will be explained.
        :param recover: Whether to recover previous computations from outfile (in case something broke during execution)
        :param idf_path: A path to the MoverScore idf files that should be used
        :param outfile: A path the output explanations should be saved to after each metric that was attacked
                        This file is used by the evaluate_attacks.py script in xai/evaluation
        :return: Currently this just returns an empty list. I could package the results in json, in the future
        '''

        # Note that that this method might throw an error when explained with the "explain_corpus" util
        # This is not problematic, as the important part is, that an outfile is generated before the results are printed
        # This outfile is analyzed by the evaluation tools

        # Here the outfile is very important, as I don't save the explanations as JSON
        explanations = self.MW.apply_hyp_explainer(df, self.explain, metrics=metrics, recover=recover,
                                                   corpus_path=None,
                                                   idf_path=None, idf_dump_path=idf_path,
                                                   precompute=lambda x: {}, outfile=explanation_path,
                                                   precomp_path=None)

        # Unpacking the explanation object
        print(explanations)
        attributions = []
        for sample in explanations:
            for key, value in sample['metrics'].items():
                print('\n\n\n\n\n')
                print("---------------------------------")
                print(key, ": ")
                for attack in self.attack_types:
                    print(value[0][attack].__str__(color_method='ansi'))
                print("src:", sample['src'])
                print("ref:", sample['ref'])
                print("orig_score:", value[1])

        # In the future I can build a real return string for this explaine
        return attributions


if __name__ == '__main__':
    # Copie
    # 3d from https://www.tensorflow.org/guide/gpu
    # To reduce memory used by tensorflow, e.g. for BERT-attack
    import tensorflow as tf

    #gpus = tf.config.list_physical_devices('GPU')
    #if gpus:
    #    try:
    #        # Currently, memory growth needs to be the same across GPUs
    #        for gpu in gpus:
    #            tf.config.experimental.set_memory_growth(gpu, True)
    #        logical_gpus = tf.config.list_logical_devices('GPU')
    #        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    #   except RuntimeError as e:
    #        # Memory growth must be set before GPUs have been initialized
    #        print(e)

    AE = AdversarialExplainer(binning_mode = 'corpus', bins = 3, attack_types = ['textfooler_adjusted'], target = 0,
                              corpus=os.path.join(ROOT_DIR,'metrics/outputs/wmt/wmt19_full_scores_raw_de_en'))

    # Creates a dill file with an attack object attacked sample saved as dill
    explain_corpus(AE,
                   recover=True,
                   from_row=0, to_row=1200,
                   outfile='attack_attributions',
                   mlqe_pandas_path=os.path.join(ROOT_DIR,'metrics/corpora/pandas_corpora/wmt/wmt19_classes/2_class_de_en'),
                   explanation_path=os.path.join(ROOT_DIR,'xai/output/explanation_checkpoints/adv_attack_de_en_2_to_0_textfooler_adjusted_wmt19.dill'))
