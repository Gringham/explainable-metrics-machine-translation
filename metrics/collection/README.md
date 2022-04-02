# Metric Collection
This folder contains wrappers for machine translation evaluation metrics. Each wrapper is implemented as a class.
E.g. `BertScore.py`. Each wrapper implements the abstract class `MetricClass`, which has two methods. 
The `__call__(ref,hyp)` method implements the scoring functionality, assuming that a list of ref and a list of hyp 
sentences is passed. As this implements the `__call__` function, it can be called by calling an instantiated object. I.e. 
`m = Bleu(); score = m(ref,hyp)`. Instead of passing the references, some metrics implement this function using the
source or using both. Additionally, some metrics, such as XMoverScore require a language pair as additional input.
The second function of the `MetricClass` is `get_abstraction(self, ref)`, which wraps the call function, such that it 
only depends on the hypothesis, while the rest of the input remains fixed. To explain, let's consider the SentenceBleu 
 variant in `Bleu.py`:

```
from metrics.collection.SacreBleu import SacreBleu

print(SacreBleu.name, SacreBleu.ref_based)
metric = SacreBleu()

# Usual metric call
ref = ["A test sentence"]
hyp = ["A simple sentence for test"]
print(metric(ref, hyp))

# Returns [0.07071067811865477]

# Fix the reference and evaluate multiple hypotheses:
metric_fixed_ref = metric.get_abstraction(ref[0])
other_hypotheses = ["A simple sentence for test", "Another simple sentence for test", 'A test sentence']
print(metric_fixed_ref(other_hypotheses))

# Returns [0.07071067811865477, 0.06389431042462725, 0.5623413251903491]
```

All of the wrappers implemented in this directory follow the same structure, though some of them, e.g. the one for Comet
implement additional parameters. During evaluation, it can be inferred, whether a metric is reference-based or reference-free
by accessing `Class.ref_based`, which is a boolean. **Each wrapper contains a working example in its main block.** I.e. to
run the wrapper on a single evaluation you can run the respective script. Specific methods, for example for the MoverScore wrapper
are commented in the source code.

Additionally, the directory `metric_libs` contains modified versions of some of the metrics. These are detailed in the README files
in their respective folders.

## Metric Wrapper
The class `MetricWrapper` implements the main evaluation loops of the project. During initialisation it creates a dictionary
that maps to all implemented metric classes. The evaluation loops look up the required classes in this dictionary and instantiate
metric objects where needed. The first loop `evaluate` takes a pandas dataframe with hyp, src, ref as input and adds a column
with the respective scores for each metric. Here is a short example:
```
import pandas as pd
from metrics.collection.MetricWrapper import MetricWrapper

metric_wrapper = MetricWrapper()
test_df = pd.DataFrame(['de-en', 'Ein cooler Satz', "A cool sentence", "Even more cool sentences", 'Dummy'],
                       index=['LP', 'SRC', 'REF', 'HYP', 'SYSTEM']).T
metric_wrapper.evaluate(test_df, outfile='test.tsv',
                        idf_path='D:\\Users\\Chris\\Desktop\Explainable_MT_Metrics_Clean\\metrics\\collection\\metrics_libs\\moverscore\\idfs\\wmt_17_idf.dill',
                        idf_dump_path='D:\\Users\\Chris\\Desktop\Explainable_MT_Metrics_Clean\\metrics\\collection\\metrics_libs\\moverscore\\idfs\\wmt17_idf.dill')

# LP	SRC	REF	HYP	SYSTEM	BERTSCORE	BLEU	BLEURT	COMET	LABSE	METEOR	MOVERSCORE	PSEUDO1	PSEUDO2	PSEUDO3	SACREBLEU	SENTCHRF	TRANSLATIONBLEU	TRANSLATIONMETEOR	TRANSQUEST	XLMR	xmoverscore_clp1	xmoverscore_clp2	xmoverscore_umd1	xmoverscore_umd2	xmoverscore_clp1_lm	xmoverscore_clp2_lm	xmoverscore_umd1_lm	xmoverscore_umd2_lm
# de-en	Ein cooler Satz	A cool sentence	Even more cool sentences	Dummy	0.9216386675834656	0.08034284189446518	0.05558368191123009	0.3638212978839874	0.4487772583961487	0.8243727598566307	0.5590383934799985	0.0	0.25	9	0.31947155212313627	0.7815607342283662	0.1597357760615681	0.8243727598566307	0.76220703125	0.8716460466384888	-0.05830156454872548	-0.026954664740348022	0.030619493170232825	0.01643878861372461	-0.985203858707661	-0.9538569588992836	-0.8962828009887027	-0.9104635055452109
``` 

The second loop `apply_hyp_explainer` applies arbitrary functions for every metric. The explainer function takes
 the hypothesis and the metric as input. Here is a short example: 
```
import pandas as pd
from metrics.collection.MetricWrapper import MetricWrapper

metric_wrapper = MetricWrapper()
test_df = pd.DataFrame(['de-en', 'Ein cooler Satz', "A cool sentence", "Even more cool sentences", 'Dummy'],
                       index=['LP', 'SRC', 'REF', 'HYP', 'SYSTEM']).T
# Any function that needs to evaluate the metric by running it
dummy_explainer = lambda hyp, metric: metric([hyp[0] + 'a'])
results = metric_wrapper.apply_hyp_explainer(test_df, dummy_explainer, outfile='save_location',
                        idf_path='D:\\Users\\Chris\\Desktop\Explainable_MT_Metrics_Clean\\metrics\\collection\\metrics_libs\\moverscore\\idfs\\wmt_17_idf.dill',
                        idf_dump_path='D:\\Users\\Chris\\Desktop\Explainable_MT_Metrics_Clean\\metrics\\collection\\metrics_libs\\moverscore\\idfs\\wmt17_idf.dill',
                        recover=False)
print(results)

# [{'hyp': 'Even more cool sentences', 'ref': 'A cool sentence', 'src': 'Ein cooler Satz', 'lp': 'de-en', 'metrics': {'BERTSCORE': [0.8459404110908508], 'BLEU': [0], 'BLEURT': [-1.329732060432434], 'COMET': [-1.3612353801727295], 'LABSE': [0.05393633618950844], 'METEOR': [0.0], 'MOVERSCORE': [-0.1959415046272106], 'PSEUDO1': [0.5], 'PSEUDO2': [0.0], 'PSEUDO3': [2], 'SACREBLEU': [0.0], 'SENTCHRF': [0.0], 'TRANSLATIONBLEU': [0.06766764161830635], 'TRANSLATIONMETEOR': [0.0], 'TRANSQUEST': [0.342041015625], 'XLMR': [0.4338158369064331], 'XMOVERSCORE': [-0.9799276955775023]}, 'times': {'BERTSCORE': 0.6210899353027344, 'BLEU': 0.007500171661376953, 'BLEURT': 0.4492199420928955, 'COMET': 1.1429998874664307, 'LABSE': 1.0700006484985352, 'METEOR': 2.4699959754943848, 'MOVERSCORE': 0.03150129318237305, 'PSEUDO1': 0.0, 'PSEUDO2': 0.0, 'PSEUDO3': 0.0, 'SACREBLEU': 0.0009996891021728516, 'SENTCHRF': 0.0, 'TRANSLATIONBLEU': 0.18650031089782715, 'TRANSLATIONMETEOR': 0.0004999637603759766, 'TRANSQUEST': 15.86349868774414, 'XLMR': 0.4955010414123535, 'XMOVERSCORE': 2.8495006561279297}}]

``` 

In case a computation breaks at some point, you can try to recover the process by setting recover = True. During the computation
the function will save states to outfile.