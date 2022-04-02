### WMT
Contains the corpora used in the WMT metrics shared task 2016-2020² as pandas df's. full_eval contains all sentences of 
systems that submitted to the wmt translation task. seg_eval contains all sentences that were used in segment-level evaluations.

² here are the related finding papers:

* Ondřej Bojar, Yvette Graham, and Amir Kamran. “Results of the WMT17 Metrics Shared Task”. In:
Proceedings of the Second Conference on Machine Translation. Copenhagen, Denmark: Association
for Computational Linguistics, Sept. 2017, pp. 489–513. doi: 10.18653/v1/W17-4755. url:
https://aclanthology.org/W17-4755.

* Qingsong Ma, Ondřej Bojar, and Yvette Graham. “Results of the WMT18 Metrics Shared Task: Both
characters and embeddings achieve good performance”. In: Proceedings of the Third Conference on
Machine Translation: Shared Task Papers. Belgium, Brussels: Association for Computational Linguistics,
Oct. 2018, pp. 671–688. doi: 10.18653/v1/W18-6450. url: https://aclanthology.org/
W18-6450.

* Qingsong Ma, Johnny Wei, Ondřej Bojar, and Yvette Graham. “Results of the WMT19 Metrics Shared
Task: Segment-Level and Strong MT Systems Pose Big Challenges”. In: Proceedings of the Fourth
Conference on Machine Translation (Volume 2: Shared Task Papers, Day 1). Florence, Italy: Association
for Computational Linguistics, Aug. 2019, pp. 62–90. doi: 10.18653/v1/W19-5302. url: https:
//aclanthology.org/W19-5302.

* Nitika Mathur, Johnny Wei, Markus Freitag, Qingsong Ma, and Ondřej Bojar. “Results of the WMT20
Metrics Shared Task”. In: Proceedings of the Fifth Conference on Machine Translation. Online: Association
for Computational Linguistics, Nov. 2020, pp. 688–725. url: https://aclanthology.org/2020.
wmt-1.77.

The script `download_wmt` downloads the respective sources, but some files need to be structured/moved to the right locations
afterwards. The folder structure of WMT should be as follows:
* wmt
    * wmt17
        * references
        * sources
        * system-outputs
        * DA-seglevel.csv
        * DA-syslevel.csv
    * wmt18
        * references
        * sources
        * system-outputs
        * DA-seglevel.csv
        * RR-syslevel.csv
    * wmt19
        * references
        * sources
        * system-outputs
        * DA-seglevel.csv
        * RR-syslevel.csv
    * wmt20
        * references
        * sources
        * system-outputs
        * DArr-seglevel.csv
        * DArr-syslevel.csv
        
The human references can be gathered from these locations:  
http://ufallab.ms.mff.cuni.cz/~bojar/wmt17-metrics-task-package.tgz   
http://ufallab.ms.mff.cuni.cz/~bojar/wmt18-metrics-task-package.tgz  
http://ufallab.ms.mff.cuni.cz/~bojar/wmt19-metrics-task-package.tgz   
https://github.com/WMT-Metrics-task/wmt20-metrics  

The class `WMTLoader` on the top level of the corpora folder can be used to convert the wmt corpora to pandas dataframes.
