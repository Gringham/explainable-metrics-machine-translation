# MoverScore
This directory contains a slightly modified version of the v1 MoverScore from 
https://github.com/AIPHES/emnlp19-moverscore, by:

Wei Zhao, Maxime Peyrard, Fei Liu, Yang Gao, Christian M. Meyer, and Steffen Eger. “MoverScore: Text
Generation Evaluating with Contextualized Embeddings and Earth Mover Distance”. In: Proceedings
of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International
Joint Conference on Natural Language Processing (EMNLP-IJCNLP). Hong Kong, China: Association
for Computational Linguistics, Nov. 2019, pp. 563–578. doi: 10.18653/v1/D19-1053. url:
https://www.aclweb.org/anthology/D19-1053.

I use the original moverscore.py, as the MoverScore documentation says that it was used to produce the original results.
(Therefore I accept a potential slowdown). In my changes I wrap MoverScore into a class, such that the model is only 
loaded once an object is instantiated. This is more efficient in my evaluation loops.
The usage is as follows:
```
from metrics.collection.metrics_libs.moverscore.moverscore import get_idf_dict, MoverScoreBase
ms = MoverScoreBase()

hyp = ["I have two dogs."]
ref = ["I have no cat and two goldfish."]

idf_dict_hyp = get_idf_dict(hyp)
idf_dict_ref = get_idf_dict(ref)

print(ms.word_mover_score(ref, hyp, idf_dict_ref, idf_dict_hyp, stop_words=[], n_gram=2))

# returns [1.0] due to idfs
```

Further, to save computing time, I precomputed a number of idf dicts and saved them to the idfs folder.
The precomputation is done with the MoverScore wrapper class. Usage of the precomputed dicts works as follows:
```
import dill
from metrics.collection.metrics_libs.moverscore.moverscore import get_idf_dict, MoverScoreBase
ms = MoverScoreBase()

hyp = ["I have two dogs."]
ref = ["I have no cat and two goldfish."]

with open('idfs/wmt17_idf.dill', 'rb') as in_strm:
    idf_dicts = dill.load(in_strm) 

# (there are also prebuildt idfs of the systems included) 
idf_dict_hyp = idf_dicts['de-en']['ref_idf']
idf_dict_ref = idf_dicts['de-en']['ref_idf']

print(ms.word_mover_score(ref, hyp, idf_dict_ref, idf_dict_hyp, stop_words=[], n_gram=2))

# returns [0.2086]
```

For the experiments with MLQE, I also use the WMT17 ref idfs.