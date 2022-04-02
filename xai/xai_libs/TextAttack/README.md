This directory contains a gpu accellerated version of the subword search that is conducted in bert-attack. Further it caps
the computation if too many permutations would have to be tested. E.g. with 7 subwords and a vocabulary of 50000 the
number of possible combinations is already 50000^7. Additionally, it does not check whether the final wordcombination 
is a real word, as the probability for it is assumed to be high, if the probability combination is high.

bert_attack_li_2020.py is modified from https://textattack.readthedocs.io/en/latest/_modules/textattack/attack_recipes/bert_attack_li_2020.html.
word_swap_masked_lm.py is modified from https://github.com/QData/TextAttack/blob/master/textattack/transformations/word_swaps/word_swap_masked_lm.py.
