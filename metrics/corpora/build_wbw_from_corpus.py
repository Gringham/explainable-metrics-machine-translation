import os

from metrics.collection.TranslationMeteor import TranslationMeteor
import pandas as pd
import json

# Short script that builds a word-by word dict from a pandas corpus
from project_root import ROOT_DIR

t = TranslationMeteor()

df = pd.read_csv(os.path.join(ROOT_DIR,'metrics/corpora/pandas_corpora/eval4nlp_test_de-zh.tsv', delimiter='\t'))

wbw = t.build_word_by_word_dict(df['SRC'], 'de-zh')
print(wbw)

json.dump(wbw, open( "de_zh_dict_eval4nlp_test.json", 'w' ) )