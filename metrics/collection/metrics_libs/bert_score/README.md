# Word-Level Bertscore
This directory contains a modified version of the BERTScore code from https://github.com/Tiiiger/bert_score, which was written by:

Tianyi Zhang, Varsha Kishore, Felix Wu, Kilian Q. Weinberger, and Yoav Artzi. “BERTScore: Evaluating
Text Generation with BERT”. In: International Conference on Learning Representations. 2020. url:
https://openreview.net/forum?id=SkeHuCVFDr.

My changes retain the token-level similarity that is used in the computation process of BERTScore. The scorer returns
A tuple of P_BERT, R_BERT and F_BERT scores, a list of reference token importance scores and a list of source token importance 
scores. Each word importance score consists of a tuple with the score and the token. Here is an example:
```
from metrics.collection.metrics_libs.bert_score.scorer import BERTScorer
custom_scorer = BERTScorer(lang='en')

hyp = ["I have two dogs."]
ref = ["I have a cat and two goldfish."]
print(custom_scorer.score(hyp, ref, verbose=True))

# ((tensor([0.9689]), tensor([0.9321]), tensor([0.9501])),\
# [[(0.9993336796760559, '<s>'), (0.9955809116363525, 'ĠI'), (0.9889905452728271, 'Ġhave'),\
# (0.9549674391746521, 'Ġtwo'), (0.9048808813095093, 'Ġdogs'), (0.9999808073043823, '.'), (0.999518871307373, '</s>')]], 
# [[(0.9993336796760559, '<s>'), (0.9955809116363525, 'ĠI'), (0.9889905452728271, 'Ġhave'), (0.9322092533111572, 'Ġa'), 
# (0.9048808813095093, 'Ġcat'), (0.8851274251937866, 'Ġand'), (0.9549674391746521, 'Ġtwo'), (0.855670690536499, 'Ġgold'), 
# (0.8715567588806152, 'fish'), (0.9999808073043823, '.'), (0.999518871307373, '</s>')]])
```

Like the normal BERTScore, it can be used in a reference-free setting, when the src is used instead of the ref and a reference-free
model is specified. The construction of word-level scores is handled separately in the explainers and evaluation functions.

Further we added the following models to utils.py:
```
    "joeddav/xlm-roberta-large-xnli"
    "Unbabel/xlm-roberta-comet-small"
    "TransQuest/monotransquest-da-multilingual" 
    "TransQuest/monotransquest-da-et_en-wiki"
    "vicgalle/xlm-roberta-large-xnli-anli"
    "cardiffnlp/twitter-xlm-roberta-base"
    "alon-albalak/xlm-roberta-large-xquad"
    "deepset/xlm-roberta-large-squad2"
```

 