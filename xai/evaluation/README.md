This folder contains evaluation scripts for explainability techniques. `eval4nlp_evaluate.py` 
contains evaluation scripts for auc, rtopk, ap and aopc. `evaluate_attacks.py` contains evaluation 
scripts for adversarial attacks. 

The `eval4nlp_evaluate.py` script is adapted from https://github.com/eval4nlp/SharedTask2021/blob/main/scripts/evaluate.py

by the organizers of the Eval4NLP shared task:

Marina Fomicheva, Piyawat Lertvittayakumjorn, Wei Zhao, Steffen Eger, and Yang Gao. “The Eval4NLP
Shared Task on Explainable Quality Estimation: Overview and Results”. In: Proceedings of the 2nd
Workshop on Evaluation and Comparison of NLP Systems. 2021. 

Here I present a number of usage examples:
### AUC, AP, RtopK
Imagine, we have explained 100 samples with the LimeExplainer and saved the results to `xai/outout/explanations/0_99_mlqe_et_attributions_lime.json` (Note
that the score for comet in this file is incorrect).
Additionally, we have a pandas dataframe that contains the ground truth data at `metrics/corpora/pandas_corpora/mlqe/mlqe_et_zeros_dropped.tsv`.
In order to evaluate the explanations with the first 100 samples of this dataframe, we can call the following function:
```
import os
from project_root import ROOT_DIR
from xai.evaluation.eval4nlp_evaluate import evaluate_mlqe_auc

evaluate_mlqe_auc([os.path.join(ROOT_DIR,'xai/output/explanations/0_99_mlqe_et_attributions_lime.json')],
                      start=0, end=100, invert=True, f=100,
                      corpus_path=os.path.join(ROOT_DIR,'metrics/corpora/pandas_corpora/mlqe/mlqe_et_zeros_dropped.tsv'))
```
This function can be used to evaluate with any other word-level error annotated corpus too. The current version
doesn't have an automatic save to a file implemented, but it can be easily added by saving each resulting dataframe.

### Evaluate Attacks
To evaluate explanations/attacks that were created with the adversarial attackers/explainers please have a look at the script
`evaluate_attacks.py`. It contains code for evaluation of success rate, perturbation rate, average introduced grammatical error
and semantic similarity. To only achieve specific goals, some parts might need to be commented out. This is described in the
comments of the script. 