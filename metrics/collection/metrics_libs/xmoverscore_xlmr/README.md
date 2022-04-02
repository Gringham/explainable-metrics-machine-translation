This directory contains a modified version of XMoverScore from 
https://github.com/AIPHES/ACL20-Reference-Free-MT-Evaluation, by:

Wei Zhao, Goran Glavaš, Maxime Peyrard, Yang Gao, Robert West, and Steffen Eger. “On the Lim-
itations of Cross-lingual Encoders as Exposed by Reference-Free Machine Translation Evaluation”.
In: Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics. Online:
Association for Computational Linguistics, July 2020, pp. 1656–1671. url: https://www.aclweb.
org/anthology/2020.acl-main.151

Like the other modified version in the xmoverscore folder, this version returns word-level scores.
Additionally, it leverages XLMR instead of mBERT. Hence, respective mappings were trained using the scripts
from the XMS github linked above. There is a slight change to the training script of mapping that approximates the
svd computation. Thereby we are able to train on 30k sentences instead of 2k.
This version keeps subwords and punctuation by default. For XLMR we provide versions of score_utils.py and score_utils_v2.py from the XMS
repo. The score utils v2 seems to be a reworked version of xms.


Example usage is as follows (to be run in the same path as the scorer.py):
```
from metrics.collection.metrics_libs.xmoverscore_xlmr.scorer import XMOVERScorer
import os, torch, truecase
from mosestokenizer import MosesDetokenizer
import numpy as np

hyp = ["I have two dogs."]
src = ["Mul pole kassi ja kahte kuldkala."]

# The xlmr version does have a parameter drop_punctuation too, however it is a dummy for compatibility reasons.
scorer = XMOVERScorer('xlm-roberta-large', 'gpt2', do_lower_case=False)

# The punctuation will be dropped during copmutation but kept with the tokens.

bam = np.loadtxt(os.path.join('mapping', 'europarl-v7.et-en.30k.17.BAM.map_base'))
projection = torch.tensor(bam, dtype=torch.float).to('cuda:0')

gbd = np.loadtxt(os.path.join('mapping', 'europarl-v7.et-en.30k.17.GBDD.map_base'))
bias = torch.tensor(gbd, dtype=torch.float).to('cuda:0')

# In this example we use preprocessing and Truecasing as in the xms repo. In some of the
# experiments we turn this feature of, as the input texts already are cased correctly. In the 
# evaluation of sentence-level scores we left it activated.
with MosesDetokenizer('de') as detokenize:
    src = [detokenize(sent.split(' ')) for sent in src]
with MosesDetokenizer('en') as detokenize:
    hyp = [detokenize(sent.split(' ')) for sent in hyp]

hyp = [truecase.get_true_case(sent) for sent in hyp]

# For xms, the layer paramerter is in use
print(scorer.compute_xmoverscore('CLP', projection, bias, src, hyp, bs=8, ngram=1, layer=17))

# Output (sent score, hyp imortance, src importance):
#([0.5890613921767689], 
#[[('<s>', 0.0899801254272461), ('▁I', 0.19881296157836914), ('▁have', 0.23381972312927246), 
#('▁two', 0.20362162590026855), ('▁dog', 0.36214756965637207), ('s', 0.1883789300918579), 
#('.', 0.03457188606262207), ('</s>', 0.03320932388305664)]], 
#[[('<s>', 0.0899801254272461), ('▁Mul', 0.3671584129333496), ('▁pole', 0.33600926399230957), 
#('▁kas', 0.29993224143981934), ('si', 0.1883789300918579), ('▁ja', 0.2523857355117798), 
#('▁kah', 0.20362162590026855), ('te', 0.3640638589859009), ('▁kul', 0.4834696054458618), ('d', 0.46680450439453125), 
#('kala', 0.39332127571105957), ('.', 0.03457188606262207), ('</s>', 0.03320932388305664)]])
```

Again, the 2-gram version is currently not supported, but earlier experiments did not lead to better results. 

To switch to the newer score_utils version provided by the XMS repository, please change the import in scorer.py:
```
#from metrics.collection.metrics_libs.xmoverscore_xlmr.score_utils import word_mover_score, lm_perplexity
-->from metrics.collection.metrics_libs.xmoverscore_xlmr.score_utils_v2 import word_mover_score, lm_perplexity
```

In this case, it returns the same scores though (i.e. with punctuation and subwords).