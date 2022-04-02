### Explainer
On top level, this directory contains various Explainer classes, that implement different Explainability methods for 
machine translation metrics. The directory contains the following subfolders (with own readmes):

* `evaluation`: Contains scripts to evaluate the explanations that were created with the Explainer Classes
* `output`: Contains explanations created with the Explainer classes and results of their evaluations. Additionally 
it contains states that were saved during computation.
* `util`: Utility to apply Explainers
* `xai_libs`: Libraries used by the Explainer Classes. Contains some additional experiments.

Earlier experiments like the ANCHOR explanations for metrics are not included in this repo. But can be added if it 
would be interesting

Further it contains the following Explainer classes (usage examples and documentation can be found in the comments and main blocks of the Explainers):

* `AdversarialAttackExplainer.py`: Applies TextAttack attacks on metrics (e.g. bert-attack, textfooler, ...)
* `ShapExplainer.py`: Using SHAP to produce feature importance scores for the metrics.
