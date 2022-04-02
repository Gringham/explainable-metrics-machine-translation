import pandas as pd
import os
import csv

from project_root import ROOT_DIR


def load_data(path):
    # A reader for the eval4nlp test set. Ref and other columns are dummies, as no ground truth is given
    src = pd.read_csv(os.path.join(path, 'test21.src'),
                     delimiter='\t', encoding='utf-8', names=['SRC'], error_bad_lines=False, quoting=csv.QUOTE_NONE)
    hyp = pd.read_csv(os.path.join(path, 'test21.mt'),
                       delimiter='\t', encoding='utf-8', names=['HYP'], error_bad_lines=False, quoting=csv.QUOTE_NONE)
    return pd.concat([src, hyp], axis=1)

if __name__ == '__main__':
    lp = 'de-zh'
    path = os.path.join(ROOT_DIR,'metrics/corpora/eval4nlp_test/'+lp+'-test21')

    df = load_data(path)
    df['LP'] = [lp]*len(df)
    df['REF'] = 'dummy'
    df['SYSTEM'] = 'dummy'
    df['DA'] = 'dummy'
    df['TAGS_HYP'] = 'dummy'


    df.to_csv(os.path.join(ROOT_DIR,'metrics/corpora/pandas_corpora/eval4nlp_test_'+lp+'.tsv', sep='\t'))