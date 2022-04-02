import os

import pandas as pd
import json

from metrics.corpora.MLQELoader import MLQELoader
from metrics.corpora.WMTLoader import WMTLoader
from project_root import ROOT_DIR


def explain_corpus(explainer,
                    from_row=0,
                    to_row=100,
                    outfile='explanations.json',
                    outpath='output\\explanations\\',
                    metrics=None,
                    loader=None,
                    mlqe_pandas_path=os.path.join(ROOT_DIR,'metrics/corpora/pandas_corpora/mlqe/mlqe_et_zeros_dropped.tsv'),
                    recover=True,
                    **kwargs):
    '''
    Utility function that applies an explainer on a corpus specified by path or other means
    :param explainer: An explainer object that implements the function apply_explanation
    :param from_row: Start row to explain
    :param to_row: End row to explain
    :param outfile: The file the output is written two (if it can be returned as json)
    :param outpath: The path where the outputfile is
    :param metrics: The metrics to apply
    :param loader: Which predefined corpus to choose
    :param mlqe_pandas_path: The path to an mlqe corpus
    :param recover: Whether to recover explanations
    :param kwargs: Other parameters can be passed via kwargs
    :return:
    '''


    if loader == 'WMT17':
        wmt17_pandas_path = os.path.join(ROOT_DIR,'metrics/corpora/pandas_corpora/wmt17_human.tsv')
        seg_df = pd.read_csv(wmt17_pandas_path, delimiter='\t')[from_row:to_row]
    elif loader == 'MLQE':
        mlqe = MLQELoader()
        seg_df = mlqe.corpus_df[from_row:to_row]
    elif mlqe_pandas_path:
        seg_df = pd.read_csv(mlqe_pandas_path, delimiter='\t')[from_row:to_row]
    else:
        raise Exception("Error, you need to specify either a WMT or MLQE as a loader or provide a path to a pandas corpis")

    attributions = explainer.apply_explanation(seg_df, recover=recover, metrics=metrics, **kwargs)
    counter = 0

    for a in attributions:
        a['corpus_row'] = from_row + counter
        counter += 1

    with open(outpath + str(from_row) + '_' + str(to_row - 1) + '_' + outfile, 'w', encoding='utf-8') as outfile:
        json.dump(attributions, outfile)

    print(attributions)