This directory contains a modified version of XMoverScore from 
https://github.com/AIPHES/ACL20-Reference-Free-MT-Evaluation, by:

Wei Zhao, Goran Glavaš, Maxime Peyrard, Yang Gao, Robert West, and Steffen Eger. “On the Lim-
itations of Cross-lingual Encoders as Exposed by Reference-Free Machine Translation Evaluation”.
In: Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics. Online:
Association for Computational Linguistics, July 2020, pp. 1656–1671. url: https://www.aclweb.
org/anthology/2020.acl-main.151

The mappings also stem from their repo.

I use the original score_utils.py file and extract additional token-level scores
by using the minimal distance per subword/word. Example usage is as follows (to be run in the 
same path as the scorer.py):
```
from metrics.collection.metrics_libs.xmoverscore.scorer import XMOVERScorer
import os, torch, truecase
from mosestokenizer import MosesDetokenizer
import numpy as np

hyp = ["I have two dogs."]
src = ["Ich habe keine Katze und zwei Goldfische"]
scorer = XMOVERScorer('bert-base-multilingual-cased', 'gpt2', do_lower_case=False, drop_punctuation=True)
# The punctuation will be dropped during copmutation but kept with the tokens.

bam = np.loadtxt(os.path.join('mapping', 'europarl-v7.de-en.2k.12.BAM.map'))
projection = torch.tensor(bam, dtype=torch.float).to('cuda:0')

gbd = np.loadtxt(os.path.join('mapping', 'europarl-v7.de-en.2k.12.GBDD.map'))
bias = torch.tensor(gbd, dtype=torch.float).to('cuda:0')

# In this example we use preprocessing and Truecasing as in the xms repo. In some of the
# experiments we turn this feature of, as the input texts already are cased correctly. In the 
# evaluation of sentence-level scores we left it activated.
with MosesDetokenizer('de') as detokenize:
    src = [detokenize(sent.split(' ')) for sent in src]
with MosesDetokenizer('en') as detokenize:
    hyp = [detokenize(sent.split(' ')) for sent in hyp]

hyp = [truecase.get_true_case(sent) for sent in hyp]

print(scorer.compute_xmoverscore('CLP', projection, bias, src, hyp, bs=8, ngram=1, layer='dummy'))

# Output:
#([-0.05474530894009466], 
#[[('I', 0.712262749671936), ('have', 0.7782429456710815), ('two', 0.8239597082138062),
# ('dogs.', 1.1226844787597656)]], 
#[[('Ich', 0.712262749671936), ('habe', 0.7782429456710815), ('keine', 0.9696831703186035), 
#('Katze', 1.1226844787597656), ('und', 1.105388879776001), ('zwei', 0.8239597082138062), 
#('Goldfische', 1.312347173690796)]])

```

I experimented with averaged word-importances per 2-gram, which did not return better results. 
In the current version 2-grams do not work on token level. 
In case you use xms with 2-grams, only use the returned sentence-score. If required,
I can fix the 2-gram word-level. 

Here another example, where drop_punctuation=False. I.e. all subwords and punctuation will be treated
separately, such that each can receive a single word-score:
```
from metrics.collection.metrics_libs.xmoverscore.scorer import XMOVERScorer
import os, torch, truecase
from mosestokenizer import MosesDetokenizer
import numpy as np

hyp = ["I have two dogs."]
src = ["Ich habe keine Katze und zwei Goldfische"]
scorer = XMOVERScorer('bert-base-multilingual-cased', 'gpt2', do_lower_case=False, drop_punctuation=False)

bam = np.loadtxt(os.path.join('mapping', 'europarl-v7.de-en.2k.12.BAM.map'))
projection = torch.tensor(bam, dtype=torch.float).to('cuda:0')

gbd = np.loadtxt(os.path.join('mapping', 'europarl-v7.de-en.2k.12.GBDD.map'))
bias = torch.tensor(gbd, dtype=torch.float).to('cuda:0')

# In this example we use preprocessing and Truecasing as in the xms repo. In some of the
# experiments we turn this feature of, as the input texts already are cased correctly. In the 
# evaluation of sentence-level scores we left it activated.
with MosesDetokenizer('de') as detokenize:
    src = [detokenize(sent.split(' ')) for sent in src]
with MosesDetokenizer('en') as detokenize:
    hyp = [detokenize(sent.split(' ')) for sent in hyp]

hyp = [truecase.get_true_case(sent) for sent in hyp]

print(scorer.compute_xmoverscore('CLP', projection, bias, src, hyp, bs=8, ngram=1, layer='dummy'))

# Output:
#([-0.10348228943269788], 
#[[('[CLS]', 0.9110100269317627), ('I', 0.712262749671936), ('have', 0.7782429456710815), ('two', 0.8239595890045166), 
#('dogs', 1.1226844787597656), ('.', 1.2364481687545776), ('[SEP]', 1.1167538166046143)]], 
#[[('[CLS]', 0.9110100269317627), ('Ich', 0.712262749671936), ('habe', 0.7782429456710815), ('keine', 0.969683051109314), 
#('Katz', 1.1226844787597656), ('##e', 1.2015368938446045), ('und', 1.105388879776001), ('zwei', 0.8239595890045166), 
#('Gold', 1.312347173690796), ('##fische', 1.3363227844238281), ('[SEP]', 1.2587172985076904)]])

```