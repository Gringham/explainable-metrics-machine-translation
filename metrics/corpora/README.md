This directory contains various corpora (please view the respective folders for a description).
The evaluation framework uses pandas. Hence, there is a number of scripts that read the
corpora into pandas dataframes. The pandas dataframes at minimum have the following columns:
* SRC
* HYP
* REF
* LP
* SYSTEM

They can also have the columns `HYP_TAGS` and `SRC_TAGS` as ground truth for word-level evaluations.
The scripts for creation of the corpora are documented inside. The following scripts are proided:
* build_wbw_from_corpus.py --> builds dicts with word by word translation
* eval4_nlp_dev_reader.py --> reads the dev sets of Eval4NLP
* eval4_nlp_dev_reader.py --> reads the test sets of Eval4NLP
* WMTLoader.py --> Reads wmt to pandas