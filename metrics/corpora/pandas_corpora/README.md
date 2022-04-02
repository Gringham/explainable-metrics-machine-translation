This directory contains tsv files of different datasets for machine translation evaluation that can be directly loaded
with pandas. Then the evaluation loops of this project can be applied. 

Specifically, data from the following corpora is included:

### Eval4NLP 
Dev and Test data of the shared task by

Marina Fomicheva, Piyawat Lertvittayakumjorn, Wei Zhao, Steffen Eger, and Yang Gao. “The Eval4NLP
Shared Task on Explainable Quality Estimation: Overview and Results”. In: Proceedings of the 2nd
Workshop on Evaluation and Comparison of NLP Systems. 2021.

I provide them as separate pandas dataframes

### WMT
These corpora need to be rebuild using the WMTLoader.py script one directory above. They are too large for github.
If requested, we can also add them to a drive location.

Contains the corpora used in the WMT metrics shared task 2016-2020² as pandas df's. full_eval contains all sentences of 
systems that submitted to the wmt translation task. seg_eval contains all sentences that were used in segment-level evaluations.


² here are the related findings:

Ondřej Bojar, Yvette Graham, and Amir Kamran. “Results of the WMT17 Metrics Shared Task”. In:
Proceedings of the Second Conference on Machine Translation. Copenhagen, Denmark: Association
for Computational Linguistics, Sept. 2017, pp. 489–513. doi: 10.18653/v1/W17-4755. url:
https://aclanthology.org/W17-4755.

Qingsong Ma, Ondřej Bojar, and Yvette Graham. “Results of the WMT18 Metrics Shared Task: Both
characters and embeddings achieve good performance”. In: Proceedings of the Third Conference on
Machine Translation: Shared Task Papers. Belgium, Brussels: Association for Computational Linguistics,
Oct. 2018, pp. 671–688. doi: 10.18653/v1/W18-6450. url: https://aclanthology.org/
W18-6450.

Qingsong Ma, Johnny Wei, Ondřej Bojar, and Yvette Graham. “Results of the WMT19 Metrics Shared
Task: Segment-Level and Strong MT Systems Pose Big Challenges”. In: Proceedings of the Fourth
Conference on Machine Translation (Volume 2: Shared Task Papers, Day 1). Florence, Italy: Association
for Computational Linguistics, Aug. 2019, pp. 62–90. doi: 10.18653/v1/W19-5302. url: https:
//aclanthology.org/W19-5302.

Nitika Mathur, Johnny Wei, Markus Freitag, Qingsong Ma, and Ondřej Bojar. “Results of the WMT20
Metrics Shared Task”. In: Proceedings of the Fifth Conference on Machine Translation. Online: Association
for Computational Linguistics, Nov. 2020, pp. 688–725. url: https://aclanthology.org/2020.
wmt-1.77.
