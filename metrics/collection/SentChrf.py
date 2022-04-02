from sacrebleu import sentence_chrf

from metrics.collection.MetricClass import MetricClass


class SentChrf(MetricClass):
    '''
    A wrapper for SentChrf from the sacrebleu library (https://github.com/mjpost/sacrebleu). CHRF was invented by:
    Maja Popović. “chrF: character n-gram F-score for automatic MT evaluation”. In: Proceedings of
    the Tenth Workshop on Statistical Machine Translation. Lisbon, Portugal: Association for Compu-
    tational Linguistics, Sept. 2015, pp. 392–395. doi: 10.18653/v1/W15-3049.
    url: https://aclanthology.org/W15-3049.
    '''
    ref_based = True
    name = 'SENTCHRF'


    def __call__(self, ref, hyp):
        '''
        :param ref: A list of strings with reference sentences
        :param hyp: A list of strings with hypothesis sentences
        :return: A list of Chrf scores per reference - hypothesis pair
        '''
        return [sentence_chrf(hypothesis=h, references=[r]).score for r, h in zip(ref, hyp)]


if __name__ == '__main__':
    b = SentChrf()

    # Sample using ref and hyp lists
    print(b(["A simple  for test"], ["A simple sentence for test"]))
    #  [0.7315483299439495]

    # Sample using a fixed reference for a list of hypothesis
    b_trimmed = b.get_abstraction("A test sentence for.")
    print(b_trimmed(["A simple sentence for test", "Another simple sentence for test", 'A test sentence for']))
    # [0.6505481100731211, 0.6046156806216636, 0.9432494159160918]
