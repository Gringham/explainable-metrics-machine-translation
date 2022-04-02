### Explainable Metrics - Machine Translation
This repository contains the source code for the experiments conducted in our paper
"Towards Explainable Evaluation Metrics for Natural Language Generation".

### Reference
If you build upon this work, we'd be happy when you cite our paper:

```
@misc{https://doi.org/10.48550/arxiv.2203.11131,
  doi = {10.48550/ARXIV.2203.11131},
  url = {https://arxiv.org/abs/2203.11131},
  author = {Leiter, Christoph and Lertvittayakumjorn, Piyawat and Fomicheva, Marina and Zhao, Wei and Gao, Yang and Eger, Steffen},
  keywords = {Computation and Language (cs.CL), Machine Learning (cs.LG), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Towards Explainable Evaluation Metrics for Natural Language Generation},
  publisher = {arXiv},
  year = {2022},
  copyright = {Creative Commons Attribution 4.0 International}
}
```

### Description
This project contains code for experiments regarding
* Hard Reference Free Metrics for Explainability
  * To apply these experiments, follow these steps:
    * Use `xai/ShapExplainer.py` to explain the metrics `TranslationMeteor` and `TranslationBleu`.
    * To apply word by word mode for TranslationMeteor, set `self.use_word_by_word=True` in `metrics/collection/TranslationMeteor.py` 
* Adversarial Attacks on Machine Translation Evaluation Metrics
  * To apply these experiments, follow these steps:
    * Use `metrics/corpora/wmt/download_wmt.py` and follow the instructions in `metrics/corpora/wmt/README.md` to prepare a wmt corpus as pandas df.
    * Use `metrics/evaluation/ProduceBins.py` to apply metrics to the corpus and determine bins for the discretization of adversarial attacks
    * Use `xai/AdversarialAttackExplainer.py` to apply attackers to the generated samples
    * Use `xai/evaluation/evaluate_attacks.py` to apply an automated evaluation
* Inverse Metrics
  * Use `xai/GreedyExplainer.py` to use the algorithm for inverse sample generation

Additionally, this project contains a skeleton to apply different explainability techniques to explain machine translation evaluation metrics.
The project is divided into two main folders *metrics* and *xai*. The *metrics* folder contains wrappers for machine translation
evaluation (MTE) metrics and evaluation loops. The *xai* folder contains a collection of Explainer classes that
explain the metrics. Further, it contains scripts that can check the generated explanations depending on their type.

For detailed information refer to the readme's and comments in the respective folders.

If files are used and edited from another projects, it is indicated in a README file at the top-level of the 
respective folder. E.g. `metrics/collection/metrics_libs/bertscore` contains libraries of BERTScore, so there 
 is a README in that folder indicating this source. Otherwise, the sources of code passages, are specified in the comments of the code.

In this main README we detail the usage of the core functionalities. At some points we explicitly set cuda,
therefore, this code will only run with a gpu.

### Installation
This project was run on Windows, therefore there might be some compatibility issues with linux. To install the packages
use the `requirements.txt` file that was build with `pipreqs`. It is recommended to use a new conda environments.
The compatibility with COMET can be a bit difficult to set up. If this occurs, please have a look at `metrics/collection/Comet.py`.
The root directory of this project can be referred to by importing `from project_root import ROOT_DIR`.


### Running Metrics
Most of the scripts in this project expect their input in terms of a pandas dataframe, where each datapoint describes a 
sample from a dataset for metric evaluation. I.e. the columns include SRC (source), HYP (hypothesis), REF (reference),
LP (language pair), SYSTEM (nmt system). Not all of the columns are required for all methods and some methods use additional
columns. Applying a metric to a dataframe means that a new column is created that contains the scores for each evaluated 
datapoint for a metric. To produce the pandas dataframes, helper scripts can be found in `metrics/corpora`.
The evaluation loop, that applies the metrics to a dataframe is implemented in `metrics/collection/MetricWrapper.py`.
When a new corpus should be used, it can simply be provided as a Pandas dataframe with the respective columns.

### Applying explanations
Explanations use a different loop that is also implemented in `metrics/collection/MetricWrapper.py`. The classes that contain
the Explainers are defined in `xai/`. They also expect a pandas dataframe that holds the input samples (with the same columns).
As a difference, the Explainers do not only evaluate the metrics, but also how the metrics will behave with perturbed input 
(or use other functionalities to explain why a metric produced a certain result). Each Explainer class contains a usage example
in its main block. In general, they can be applied with following utility function:

```
import os
from project_root import ROOT_DIR
from xai.util.corpus_explainer import explain_corpus

# Initialize the explainer object (Pseudo Code)
E = ExplainerObject()

# Apply the explainer to a corpus
explain_corpus(E,                                   # The explainer object
               recover=False,                       # Whether to recover previous explanations
               from_row=0, to_row=100,              # Which rows of a corpus to consider (default corpus are mlqe-pe et-en rows)
               outfile='attribution.json')          # The name of the output file

# After this, feature importance explanations are saved in the specified output file under xai/outputs/explanations. 
# Adversarial attacks are instead evaluated with the objects that are saved to xai/outputs/explanation_checkpoints.
``` 


### Evaluating plausibility with AUC, AP, RtopK to word-level error annotations
The evaluation with AUC, AP and RtopK is handled and described in `xai/evaluation/`. Extending the previous pseudocode,
a simple evaluation can be performed by calling:
```
# loads the previous output file that was prepended with its starting and ending row
evaluate_mlqe_auc([
        os.path.join(ROOT_DIR,'xai/output/explanations/0_99_attribution.json')], 
        start=0,                        # The start row
        end=100,                        # The end row
        invert=True,                    # Whether to invert feature importance scores
        f=20)                           # Starting increment for output filenames
```

The results are printed as pandas dataframe. The script `xai/SystemPaperScores.py` reproduces scores of the Eval4NLP system paper.


### Evaluating adversarial attacks
Adversarial attacks on metrics can be applied using the `xai/AdversarialAttackExplainer.py`. For usage, see the comments. They can be evaluated using the file 
`xai/evaluation/evaluate_attacks.py`. 

Sample usage of Bert-Attack (the parameters are described in the respective file):
```
AE = AdversarialExplainer(binning_mode = 'corpus', bins = 3, attack_types = ['bert-attack'], target = 0,
                              corpus=os.path.join(ROOT_DIR,'metrics/outputs/wmt/wmt20_full_scores_raw_zh_en'))
explain_corpus(AE,
           recover=True,
           from_row=0, to_row=1200,
           outfile='attack_attributions',
           mlqe_pandas_path=os.path.join(ROOT_DIR,'metrics/corpora/pandas_corpora/wmt/wmt20_classes/2_class_zh_en'),
           explanation_path=os.path.join(ROOT_DIR,'xai/output/explanation_checkpoints/adv_attack_zh_en_2_to_0_tf.dill'))
```
This will attack all metrics using the sentences from a corpus passed with `mlqe_pandas_path` and save explanation objects to `explanation_path`.

### Adding new metrics
In order to add a new metric to the evaluations, create a new class in `metrics/collection` and implement the class `MetricClass` (see the README.md of that folder).
You can use the other metrics as an example. Further, add the metric to the `MetricWrapper` class, to enable it for evaluations. Therefore,
add the metric to the dictionary that is defined in the beginning of the `MetricWrapper.py` script. 

### Adding new explainability techniques
In order to add a new explainability technique, create a new class in `xai` that implements `xai/Explainer`. As an example
you can also look at the existing explainbility techniques. 

### Walkthrough
Finally, we present a short walkthrough that shows the parts that were teaserd before in action. Assume we have written a 
new metric `NewScore`, that grades translations based on the length of the shortest word. I.e. if a translation contains a 
word with only one letter, it is a good translation:

```
from metrics.collection.MetricClass import MetricClass

class NewScore(MetricClass):
    ref_based = True
    name = 'NewScore'


    def __call__(self, ref, hyp):
        res = []
        for h in hyp:
            try:
                res.append(-min([len(a) for a in h.split()]))
            except:
                res.append(0.0)

        return res
``` 

Let us try out out new metric:
```
n = NewScore()
print(n(['Any Reference Sentence', 'Any Reference Sentence'], ['A sentence with a short word', 'Long wordy sentence']))

# [-1, -4]
```
As you can see our metric assigned a higher score to the sentence with the short word `A`.

Next, let us assume we have a dataset that contains human annotations on translation quality. Then we could transform this
dataset into a pandas dataframe as follows:

```
import pandas as pd

test_df = pd.DataFrame([['de-en', 'Ein cooler Satz', "A cool sentence", "Even more cool sentences", 'Dummy', 0.9],
                        ['de-en', 'Ein zweiter cooler Satz', "A second cool sentence", "Even more cool sentences two", 'Dummy', 0.3],
                        ['de-en', 'Das ist wirklich der letzte Satz', "This really is the last sentence", "This is not the last sentence", 'Dummy', 0.5]],
                       columns=['LP', 'SRC', 'REF', 'HYP', 'SYSTEM', 'DA'])
```
This dataframe should contain columns for the language pair (here German-English), for the source sentence SRC, the reference sentence REF,
the hypothesis HYP, the System SYSTEM and the human annotations DA. Depending on the metric not all of these values are necessary and 
some can be filled with dummies (for our metric SRC and REF could be dummies as well).

If we do not register our metric manually for the evaluation loops in the source code, we can do so during runtime:
```
from metrics.collection.MetricWrapper import MetricWrapper

metric_wrapper = MetricWrapper()
metric_wrapper.metrics['NEWSCORE'] = NewScore
``` 

To evaluate our metric for the pearson correlation with human judgements, we can now use the evaluation loop of the MetricWrapper:
```
from metrics.evaluation.pearson_eval import evaluate_seg
scores = metric_wrapper.evaluate(test_df, metrics = ['NEWSCORE'])

print(evaluate_seg(scores, metrics = ['NEWSCORE'], output=None, da_column='DA'))

# LP	NEWSCORE
# de-en	-0.6546536707079772
```

As you can see, our metric has a rather weak correlation with human judgements ;) 


Next, assume our dataset additionally contains ground-truth word-level error annotations (in this case non-sensical) that we want to compare our 
scores with. Therefore, lets reassign the example from before:
```
test_df = pd.DataFrame([['de-en', 'Ein cooler Satz', "A cool sentence", "Even more cool sentences", 'Dummy', 0.9, '[0, 1, 0, 0]'],
                        ['de-en', 'Ein zweiter cooler Satz', "A second cool sentence", "Even more cool sentences two", 'Dummy', 0.3, '[0, 0, 0, 0, 1]'],
                        ['de-en', 'Das ist wirklich der letzte Satz', "This really is the last sentence", "This is not the last sentence", 'Dummy', 0.5, '[0, 1, 0, 0, 0, 0]']],
                       columns=['LP', 'SRC', 'REF', 'HYP', 'SYSTEM', 'DA', 'TAGS_HYP'])
```

Then we can evaluate the AUC, AP, RtopK scores as follows:

```
from xai.util.corpus_explainer import explain_corpus
from xai.evaluation.eval4nlp_evaluate import evaluate_mlqe_auc
# Save our corpus to a file
test_df.to_csv('test_corpus.tsv', sep='\t', index=False)

# Use the explanation utils to explain directly on the file (and save the attributions to JSON)
explain_corpus(LE, metrics=['NEWSCORE'], recover=False,from_row=0, to_row=4,outfile='NewScore_le.json',outpath='',mlqe_pandas_path='test_corpus.tsv',)
explain_corpus(IM, metrics=['NEWSCORE'], recover=False,from_row=0, to_row=4,outfile='NewScore_im.json',outpath='',mlqe_pandas_path='test_corpus.tsv')

# Evaluate the resulting explanation files
evaluate_mlqe_auc(['0_3_NewScore_le.json'], start=0, end=4, invert=True, f=100, corpus_path='test_corpus.tsv')
evaluate_mlqe_auc(['0_3_NewScore_im.json'], start=0, end=4, invert=True, f=100, corpus_path='test_corpus.tsv')

# The last 6 functions will return the following results:
#     Metric  AUC     AP  REC_TOPK  PEARSON
#0  NEWSCORE  0.0  0.206       0.0   -0.655
#     Metric    AUC   AP  REC_TOPK  PEARSON
#0  NEWSCORE  0.444  0.5     0.333   -0.655
```

As a last step lets assume we want to attack our metric on the three presented samples. Therefore
we can use an attack explainer:
```
from xai.AdversarialAttackExplainer import AdversarialExplainer

AE = AdversarialExplainer(binning_mode = 'loose', bins = 5, loose_span=5, attack_types = ['bert-attack'], target = 0)
AE.MW.metrics['NEWSCORE'] = NewScore
im_explanations = AE.apply_explanation(test_df, metrics=['NEWSCORE'], explanation_path='')

# In this case no attack succeeded
# NEWSCORE : 
# 2 (100%) --> [FAILED]
# Even more cool sentences
# src: Ein cooler Satz
# ref: A cool sentence
# orig_score: -4
# ---------------------------------
# NEWSCORE : 
# 3 (100%) --> [FAILED]
# Even more cool sentences two
# src: Ein zweiter cooler Satz
# ref: A second cool sentence
# orig_score: -3
# ---------------------------------
# NEWSCORE : 
# 4 (100%) --> [FAILED]
# This is not the last sentence
# src: Das ist wirklich der letzte Satz
# ref: This really is the last sentence
# orig_score: -2
```