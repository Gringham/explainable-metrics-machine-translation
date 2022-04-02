import pandas as pd
import os
import csv
# A reader for the test data of the dev sets of the Eval4NLP shared task. It
from project_root import ROOT_DIR


def load_data(path, type='dev'):
    # Read the different files provided into single pandas df's then concatenate them
    tgt_tags = pd.read_csv(os.path.join(path, type+'.tgt-tags'),
                      delimiter='\t', encoding='utf-8', names=['TAGS_HYP'], error_bad_lines=False, quoting=csv.QUOTE_NONE)
    src_tags = pd.read_csv(os.path.join(path, type+'.src-tags'),
                           delimiter='\t', encoding='utf-8', names=['TAGS_SRC'], error_bad_lines=False,
                           quoting=csv.QUOTE_NONE)
    src = pd.read_csv(os.path.join(path, type+'.src'),
                     delimiter='\t', encoding='utf-8', names=['SRC'], error_bad_lines=False, quoting=csv.QUOTE_NONE)
    hyp = pd.read_csv(os.path.join(path, type+'.mt'),
                       delimiter='\t', encoding='utf-8', names=['HYP'], error_bad_lines=False, quoting=csv.QUOTE_NONE)
    zscores = pd.read_csv(os.path.join(path, type+'.da'),
                          delimiter='\t', encoding='utf-8', names=['DA'], error_bad_lines=False, quoting=csv.QUOTE_NONE)
    return pd.concat([src, hyp, zscores, tgt_tags, src_tags], axis=1)

if __name__ == '__main__':
    # To run change all occurences of the lp to the lp you want to have
    path = os.path.join(ROOT_DIR,'metrics/corpora/eval4nlp_test/et-en-test21')
    lp = 'et-en'

    df = load_data(path, type='test21')
    df['LP'] = [lp]*len(df)
    df['REF'] = 'dummy'
    df['SYSTEM'] = 'dummy'


    df.to_csv(os.path.join(ROOT_DIR,'metrics/corpora/pandas_corpora/eval4NLP/eval4nlp_test_et-en-gold.tsv'), sep='\t')