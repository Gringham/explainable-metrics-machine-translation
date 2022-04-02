import sacrebleu
from sacremoses import MosesTokenizer

from metrics.collection.MetricClass import MetricClass


class SacreBleu(MetricClass):
    '''A wrapper for SacreBleu SentenceBleu (https://github.com/mjpost/sacrebleu) a Sentence BLEU implementation by:
    Matt Post. “A Call for Clarity in Reporting BLEU Scores”. In: Proceedings of the Third Conference on
    Machine Translation: Research Papers. Brussels, Belgium: Association for Computational Linguistics, Oct.
    2018, pp. 186–191. doi: 10.18653/v1/W18-6319. url: https://aclanthology.org/W18-
    6319.
    '''
    ref_based=True
    name = 'SACREBLEU'


    def __init__(self):
        self.tokenizer = MosesTokenizer(lang='en')

    def __call__(self, ref, hyp):
        '''
        :param ref: A list of strings with reference sentences
        :param hyp: A list of strings with hypothesis sentences
        :return: A list of SacreBleu Sentence Bleu scores per reference - hypothesis pair
        '''

        return [sacrebleu.sentence_bleu(hyp[x], [ref[x]], smooth_method='add-k', smooth_value=1).score/100 for x in range(len(hyp))]



if __name__ == '__main__':
    b = SacreBleu()

    # Sample using ref and hyp lists
    print(b(["A simple  for test"],["A simple sentence for test"]))
    #[0.44721359549995787]

    # Sample using a fixed reference for a list of hypothesis
    b_trimmed = b.get_abstraction("A test sentence for.")
    print(b_trimmed(["A simple sentence for test", "Another simple sentence for test", 'A test sentence for']))
    #[0.40410310093532475, 0.37606030930863954, 0.7788007830714052]