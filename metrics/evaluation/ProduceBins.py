import pandas as pd
import os

from metrics.collection.MetricWrapper import MetricWrapper
from project_root import ROOT_DIR

'''
These methods are used to produce equally sized bins/classes for each metric in a dataframe that has been appended with metric scores
'''

def get_bins_from_corpus(corpus=os.path.join(ROOT_DIR,'metrics/outputs/wmt/wmt19_full_scores_raw_de_en'), n=5):
    '''
    :param path: to corpus
    :param n: number of bins
    :return: a dictionary with a key per metric. Each metric has a list of n bins
    '''
    df = pd.read_csv(corpus, delimiter='\t')

    bins = {}
    for key in df.columns:
        if pd.api.types.is_numeric_dtype(df[key]): # use binning for every numeric column
            df['a'] = pd.qcut(df[key], q=n, duplicates='drop')

            # get the bins for a metric by applying a quantile cut with n elements und finding the unique elements
            bins[key] = list(set(pd.qcut(df[key], q=n, duplicates='drop')))
            bins[key].sort()

    return bins, df

def score_wmt_19_lang(lp='de-en'):
    '''
    :param lp: language pair
    :return: dataframe with metric scores for this language pair (running the evaluation loop of the metric wrapper)
    '''
    import tensorflow as tf

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
    mw = MetricWrapper()
    path = os.path.join(ROOT_DIR,'metrics/corpora/pandas_corpora/wmt/wmt19.tsv')
    out_path = os.path.join(ROOT_DIR,'metrics/outputs/wmt/wmt19_full_scores_raw_'+lp.replace('-','_'))
    df = pd.read_csv(path, delimiter='\t')
    df['HYP'].fillna('', inplace=True)
    df = df[df['LP']==lp]

    idf_path = os.path.join(ROOT_DIR,'metrics/collection/metrics_libs/moverscore/idfs/wmt19_idf_'+lp.replace('-','_')+'.dill')
    df_scored = mw.evaluate(df, outfile=out_path, corpus_path=None,
                                     idf_path=None, idf_dump_path=idf_path, recover=True)

    return df_scored

def find_bins(df, metrics, bins, tgt = 0):
    '''This function selects all datapoints that fall into the target class for all metrics in the passed list of metrics'''
    def check_row(x):
        for metric in metrics:
            if x[metric] not in bins[metric][tgt]:
                return False
        else:
            return True

    return df[df.apply(check_row, axis=1)]






if __name__ == '__main__':
    mw = MetricWrapper()
    metrics = list(mw.metrics.keys())
    metrics += ['xmoverscore_clp2_lm']
    metrics.remove('XMOVERSCORE')

    # Remove metrics that shouldn't be computed
    metrics.remove('TRANSLATIONBLEU')
    metrics.remove('TRANSLATIONMETEOR')

    # Comment this in, when you want to produce new bins based on wmt19 (after preparing the pandas dataframe)
    # score_wmt_19_lang('de-en')
    n = 3

    '''
    For each in 3 bins, we create a small dataset where all lie in the same class and save it to a file
    '''
    for x in range(n):
        print(x)
        bins, df = get_bins_from_corpus(n = n)
        selected = find_bins(df, metrics=metrics, bins=bins, tgt=x)
        selected.reset_index(inplace=True)
        selected.to_csv(os.path.join(ROOT_DIR,"metrics/corpora/pandas_corpora/wmt/wmt19_classes/"+str(x)+"_class_de_en"), sep='\t')