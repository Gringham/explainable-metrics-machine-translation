import os
import dill
import torch

from metrics.collection.TranslationMeteor import TranslationMeteor
from metrics.collection.BertScore import BertScore
from metrics.collection.Bleurt import Bleurt
from metrics.collection.Comet import Comet
from metrics.collection.Meteor import Meteor
from metrics.collection.MoverScore import MoverScore
from metrics.collection.SacreBleu import SacreBleu
from metrics.collection.SentChrf import SentChrf
from metrics.collection.TranslationBleu import TranslationBleu
from metrics.collection.Transquest import Transquest
from metrics.collection.XMoverScore import XMoverScore

import pandas as pd
import time, tqdm

import tensorflow as tf

from project_root import ROOT_DIR


class MetricWrapper:
    def __init__(self):
        # A dictionary to all implemented metric wrappers
        self.metrics = {
            'BERTSCORE': BertScore,
            'BLEURT': Bleurt,
            'COMET': Comet,
            'METEOR': Meteor,
            'MOVERSCORE': MoverScore,
            'SACREBLEU': SacreBleu,
            'SENTCHRF': SentChrf,
            'TRANSLATIONBLEU': TranslationBleu,
            'TRANSLATIONMETEOR': TranslationMeteor,
            'TRANSQUEST': Transquest,
            'XMOVERSCORE': XMoverScore
        }

    def evaluate(self, samples_df, metrics=None, outfile=None, print_time=True,
                 idf_path=os.path.join(ROOT_DIR,'metrics/collection/metrics_libs/moverscore/idfs/idf.dill'),
                 corpus_path=os.path.join(ROOT_DIR,'metrics/corpora/pandas_corpora/eval4NLP/eval4mlp_test_de-zh.tsv'),
                 idf_dump_path=os.path.join(ROOT_DIR,'metrics/collection/metrics_libs/moverscore/idfs/idf.dill'),
                 x_mover_mappings=os.path.join(ROOT_DIR,'metrics/collection/metrics_libs/xmoverscore/mapping'),
                 hyp_column='HYP',
                 system_column='SYSTEM',
                 metric_postfix='',
                 recover=False):
        '''
        Runs all metrics specified in the list of metrics on a given dataframe
        :param samples_df: A pandas df with at least the following columns - LP, SRC, REF, HYP, SYSTEM
                           if you only evaluate one system just place some dummy value for SYSTEM. LP should be a language
                           pair separated by '-'
        :param metrics: A list of metrics that should be evaluated
        :param outfile: A file where the scores are written to as tsv
        :param print_time: If true, the complete runtime for each metric is printed out
        :param idf_path: If specified, precomputed idfs are loaded for moverscore. The precomputation can be run by setting
        this parameter to None and setting the corpus_path to a tsv file containing the corpus (produced by WMT loader).
        :param corpus_path: If idf_path is not specified, this path will be used by moverscore to load a corpus from a tsv.
        If both are set to None, the dataframe passed to this function will be used.
        :param idf_dump_path: A path where precalculated moverscore idfs can be saved.
        :param x_mover_mappings: A path to the mappings used for xmoverscore. This parameter has no effect if the xlm version
                                 of xms is used
        :param hyp_column: The column name of the column with the hypotheses
        :param metric_postfix: The prefix of the column that is created for each metric
        :param recover: If true, existing columns with metric names will be skipped, as they have already been created
        :return: dataframe with metric scores per column
        '''

        # Sort by language pair and system
        sorted_input = samples_df.sort_values(by=['LP', system_column])
        sorted_input.reset_index(drop=True, inplace=True)
        lp_list = sorted_input['LP'].unique().tolist()

        # get lists of ref, hyp and src sentences
        ref_list = sorted_input['REF'].tolist()
        hyp_list = sorted_input[hyp_column].tolist()
        src_list = sorted_input['SRC'].tolist()

        # if no metrics are specified all metrics are used
        if not metrics:
            metrics = self.metrics.keys()

        pre_translations = None
        for key in metrics:
            print(key, torch.cuda.memory_allocated() , torch.cuda.max_memory_allocated())
            if recover:  # Skip if metric column already exists in dataframe and recover = True
                if key + metric_postfix in samples_df.columns:
                    continue
            s = time.time()
            metric = self.metrics[key]()
            print(key, torch.cuda.memory_allocated(), torch.cuda.max_memory_allocated())
            # The translation metrics require translations of the src, which can be precomputed for efficiency

            # Use this if something failed after precomputing translations
            #with open("translation_save,txt", 'rb') as pickle_file:
            #    pre_translations = dill.load(pickle_file)
            if (key == 'TRANSLATIONBLEU' or key == 'TRANSLATIONMETEOR' or key == 'TRANSLATIONCOMET') and pre_translations == None:
                print("precomputing translations")
                pre_translations = {}
                for lp in set(lp_list):
                    pre_translations[lp] = metric.precompute_translations(src_list, lp)
                try:
                    with open("translation_save,txt", 'wb') as pickle_file: #save the precomputed values to a file
                        # Dill provides more options than pickle
                        dill.dump(pre_translations, pickle_file, -1)
                except Exception as e:
                    print("Couldn't bew saved", str(e))

            # Comet needs both, src and ref
            if key == 'COMET':
                sorted_input[key + metric_postfix] = metric(src_list, ref_list, hyp_list)
                del metric.model
                torch.cuda.empty_cache()


            elif key == 'MOVERSCORE':
                # A dict of systems per lang pair, necessary for moverscore idfs
                scores = []

                # Detokenize the corpus as preprocessing, similar to the preprocessing done for run_mt
                tokenized_input = metric.preprocess_df(sorted_input, hyp_column)

                # Get a dictionary of systems per language pair
                lp_system_dict = {lp: tokenized_input[tokenized_input['LP'] == lp][system_column].unique().tolist() for
                                  lp in
                                  lp_list}

                # As done in the original moverscore repo the idfs are built from a corpus
                # They can either be loaded precomputed loading with dill (like pickle)
                # or alternatively they can be created on the fly (takes some time to run)
                if (idf_path):  # load from path
                    idfs = metric.load_idf_dict_from_file(idf_path)
                elif corpus_path:  # load corpus and compute from there
                    idfs = metric.gen_idf_from_corpus(corpus_path, idf_dump_path, hyp_column=hyp_column,
                                                      system_column=system_column)
                else:  # use the current input dataframe
                    idfs = metric.gen_idf_from_corpus(samples_df, idf_dump_path, hyp_column=hyp_column,
                                                      system_column=system_column)

                for lp in tqdm.tqdm(lp_list, desc='MoverScore LP progress'):
                    # Select a language pair
                    filteref_df = tokenized_input[tokenized_input['LP'] == lp][['REF', hyp_column, system_column]]
                    ref_idf = idfs[lp]['ref_idf']

                    for sys in tqdm.tqdm(lp_system_dict[lp], desc='MoverScore System progress'):
                        # iterate all systems
                        sys_filter = filteref_df[filteref_df[system_column] == sys][['REF', hyp_column]]
                        ref = sys_filter['REF'].tolist()
                        hyp = sys_filter[hyp_column].tolist()
                        # If the system entry can be found in the idf dict, we will use it. Else, we will use the ref idfs
                        try:
                            hyp_idf = idfs[lp]['hyp_idfs'][sys]
                        except:
                            hyp_idf = ref_idf
                        scores += metric(ref, hyp, ref_idf, hyp_idf)
                sorted_input[key + metric_postfix] = scores
                del metric.ms.model
                torch.cuda.empty_cache()

            elif key == 'XMOVERSCORE':
                # we return 8 scores for xms
                scores = [[]] * 8
                for lp in lp_list:
                    filteref_df = sorted_input[sorted_input['LP'] == lp][['SRC', hyp_column]]
                    src = filteref_df['SRC'].tolist()
                    hyp = filteref_df[hyp_column].tolist()
                    lp_results = metric(src, hyp, lp=lp, mapping_path=x_mover_mappings)
                    for x in range(len(lp_results)):
                        if x % 2 == 0:  # every second score is one without lm but src and hyp token scores
                            scores[x] = scores[x] + lp_results[x][0]
                        else:
                            scores[x] = scores[x] + lp_results[x]
                sorted_input['xmoverscore_clp1' + metric_postfix] = scores[0]
                sorted_input['xmoverscore_clp2' + metric_postfix] = scores[2]
                sorted_input['xmoverscore_umd1' + metric_postfix] = scores[4]
                sorted_input['xmoverscore_umd2' + metric_postfix] = scores[6]
                sorted_input['xmoverscore_clp1_lm' + metric_postfix] = scores[1]
                sorted_input['xmoverscore_clp2_lm' + metric_postfix] = scores[3]
                sorted_input['xmoverscore_umd1_lm' + metric_postfix] = scores[5]
                sorted_input['xmoverscore_umd2_lm' + metric_postfix] = scores[7]

            elif key == 'TRANSQUEST' or key == 'TRANSLATIONBLEU' or key == 'TRANSLATIONMETEOR':
                # All these need slightly different parameters
                res = []
                for lp in lp_list:
                    filteref_df = sorted_input[sorted_input['LP'] == lp][['SRC', hyp_column]]
                    src = filteref_df['SRC'].tolist()
                    hyp = filteref_df[hyp_column].tolist()
                    if key == 'TRANSLATIONBLEU' or key == 'TRANSLATIONMETEOR':
                        res += metric(src, hyp, lp=lp, preprocess=True, pre_translations=pre_translations[lp])
                    else:
                        res += metric(src, hyp, lp=lp)
                sorted_input[key + metric_postfix] = res

            elif self.metrics[key].ref_based:
                # all other ref based metrics can be handled together
                sorted_input[key + metric_postfix] = metric(ref_list, hyp_list)

            else:
                # all other src based metrics can be handled together
                sorted_input[key + metric_postfix] = metric(src_list, hyp_list)

            if outfile:
                sorted_input.to_csv(outfile, sep='\t', index=False)
            self.clean_tensor_mem()

            if print_time:
                print(key, 'Used Time', time.time() - s)


            del metric


        return sorted_input

    def apply_hyp_explainer(self, samples_df, explainer_fn, metrics=None,
                            outfile=os.path.join(ROOT_DIR,'xai/output/explanation_checkpoints/explanation_objects.dill'),
                            print_time=True,
                            idf_path=os.path.join(ROOT_DIR, 'metrics/collection/metrics_libs/moverscore/idfs/idf.dill'),
                            corpus_path=os.path.join(ROOT_DIR,
                                                     'metrics/corpora/pandas_corpora/eval4NLP/eval4mlp_test_de-zh.tsv'),
                            idf_dump_path=os.path.join(ROOT_DIR,
                                                       'metrics/collection/metrics_libs/moverscore/idfs/idf.dill'),
                            x_mover_mappings=os.path.join(ROOT_DIR,'metrics/collection/metrics_libs/xmoverscore/mapping'),
                            precomp_path=os.path.join(ROOT_DIR,'xai/output/explanation_checkpoints/saved_precomp.dill'),
                            precompute=None,
                            recover=False):
        sorted_input = samples_df.sort_values(by=['LP', 'SYSTEM'])
        lp_list = sorted_input['LP'].tolist()

        ref_list = sorted_input['REF'].tolist()
        hyp_list = sorted_input['HYP'].tolist()
        src_list = sorted_input['SRC'].tolist()

        if not metrics:
            metrics = self.metrics.keys()

        explanations = []
        if recover:
            # Recover samples from dill, assuming the same df is used
            try:
                with open(outfile, 'rb') as pickle_file:
                    explanations = dill.load(pickle_file)
                    print('Recovered Explanations for: ', list(explanations[0]['metrics'].keys()))
            except:
                # create a new dictionary with x sanmpes
                for x in tqdm.tqdm(range(0, len(hyp_list)), desc='Precompute: '):
                    sample = {'hyp': hyp_list[x], 'ref': ref_list[x], 'src': src_list[x], 'lp': lp_list[x],
                              'metrics': {}, 'times': {}}
                    explanations.append(sample)

            # Recover samples from dill, assuming the same df is used
            if precomp_path:
                with open(precomp_path, 'rb') as pickle_file:
                    pre_dicts = dill.load(pickle_file)  # load precomputed preprocessing steps
                    if 'gen' in pre_dicts:
                        # specific step to recover precomputations for custom lime
                        for l in range(len(pre_dicts)):
                            pre_dicts[l]['gen'][0] = explanations[l]['hyp']
                            pre_dicts[l]['masks'][:-1] = [[1 if m == 0 else 0 for m in mask] for mask in
                                                          pre_dicts[l]['masks'][:-1]]
                            pre_dicts[l]['masks'][0] = [1] * len(pre_dicts[l]['masks'][0])
                            pre_dicts[l]['distances'][0] = 0
                        print('Recovered Precomputations')
            else:
                if precompute: # if no file path is passed we just do the normal precompute
                    pre_dicts = []
                    for x in tqdm.tqdm(range(0, len(hyp_list)), desc='Precompute: '):
                        pre_dicts.append(precompute(hyp_list[x]))


        else:
            pre_dicts = []
            for x in tqdm.tqdm(range(0, len(hyp_list)), desc='Precompute: '):
                sample = {'hyp': hyp_list[x], 'ref': ref_list[x], 'src': src_list[x], 'lp': lp_list[x], 'metrics': {},
                          'times': {}}
                if precompute:
                    # Allows to pass custom precomputation function. E.g. precomputed pertrubations for marginalization
                    pre_dicts.append(precompute(hyp_list[x]))
                explanations.append(sample)
            if precompute and precomp_path:
                with open(precomp_path, 'wb') as pickle_file: #save the precomputed values to a file
                    # Dill provides more options than pickle
                    dill.dump(pre_dicts, pickle_file, -1)

        pre_translations = None
        #explanations[0]['metrics']['MOVERSCORE'] = None
        for key in metrics:
            if not key in explanations[0]['metrics']: # Are there already values for this metric?
                start = 0
            else:
                start = sum([1 if key in explanations[x]['metrics'] else 0 for x in range(len(explanations))])

            print('Starting at:', start)
            if start != len(hyp_list): # if not precomputed
                metric = self.metrics[key]()
                if key == 'MOVERSCORE': # load moverscore idfs
                    if idf_path:
                        idfs = metric.load_idf_dict_from_file(idf_path)
                    elif corpus_path:
                        idfs = metric.gen_idf_from_corpus(corpus_path, idf_dump_path)
                    else:
                        idfs = metric.gen_idf_from_corpus(samples_df, idf_dump_path)
                if (key == 'TRANSLATIONBLEU' or key == 'TRANSLATIONMETEOR') and pre_translations == None and len(
                    set(lp_list)) == 1:
                    print("precomputing translations")
                    pre_translations = metric.precompute_translations(src_list, lp_list[0])

            for x in tqdm.tqdm(range(start, len(hyp_list)), desc='Processing key:' + key):
                if precompute:
                    # using this as a way to pass the current key in each iteration
                    pre_dicts[x]['key'] = key
                    explainer = lambda hyp, metric: explainer_fn(hyp, metric, pre_dict=pre_dicts[x])
                else:
                    explainer = explainer_fn
                s = time.time()

                # Apply the explainer function for different types of metrics
                if key == 'COMET':
                    explanations[x]['metrics'][key] = explainer(hyp_list[x],
                                                                metric.get_abstraction(src_list[x], ref_list[x]))

                elif key == 'MOVERSCORE':
                    explanations[x]['metrics'][key] = explainer(hyp_list[x],
                                                                metric.get_abstraction(ref_list[x],
                                                                                       idfs[lp_list[x]]['ref_idf'],
                                                                                       idfs[lp_list[x]]['ref_idf']))

                elif key == 'XMOVERSCORE':
                    explanations[x]['metrics'][key] = explainer(hyp_list[x],
                                                                metric.get_abstraction(src_list[x], lp_list[x],
                                                                                       mapping_path=x_mover_mappings))

                elif key == 'TRANSLATIONBLEU' or key == 'TRANSLATIONMETEOR':
                    explanations[x]['metrics'][key] = explainer(hyp_list[x],
                                                                metric.get_abstraction(src_list[x], lp_list[x],
                                                                                       pre_translations))

                elif key == 'TRANSQUEST':
                    explanations[x]['metrics'][key] = explainer(hyp_list[x],
                                                                metric.get_abstraction(src_list[x], lp_list[x]))

                elif self.metrics[key].ref_based:
                    explanations[x]['metrics'][key] = explainer(hyp_list[x], metric.get_abstraction(ref_list[x]))

                else:
                    explanations[x]['metrics'][key] = explainer(hyp_list[x], metric.get_abstraction(src_list[x]))

                if print_time:
                    explanations[x]['times'][key] = time.time() - s
                    print(key, 'Used Time', explanations[x]['times'][key])

            self.clean_tensor_mem()

            # Save after every sample. This is saved as dill, as the explanations at this state can be various objects
            if outfile:
                with open(outfile, 'wb') as pickle_file:
                    # Dill provides more options than pickle
                    dill.dump(explanations, pickle_file, -1)

        return explanations

    def clean_tensor_mem(self):
        # Copied this part from https://stackoverflow.com/questions/41117740/tensorflow-crashes-with-cublas-status-alloc-failed
        # This fixed my memory issues with tensorflow
        # An alternative can be to run the tensorflow part in a subprocess, which would be kind of ugly
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                # Currently, memory growth needs to be the same across GPUs
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            except RuntimeError as e:
                # Memory growth must be set before GPUs have been initialized
                print(e)


if __name__ == '__main__':
    metric_wrapper = MetricWrapper()
    test_df = pd.DataFrame(['de-en', 'Ein cooler Satz', "A cool sentence", "Even more cool sentences", 'Dummy'],
                           index=['LP', 'SRC', 'REF', 'HYP', 'SYSTEM']).T
    metric_wrapper.evaluate(test_df, outfile=os.path.join(ROOT_DIR,'metrics/outputs/test.tsv'),
                            idf_path=os.path.join(ROOT_DIR,'metrics/collection/metrics_libs/moverscore/idfs/wmt_17_idf.dill'),
                            idf_dump_path=os.path.join(ROOT_DIR,'metrics/collection/metrics_libs/moverscore/idfs/wmt17_idf.dill'))
