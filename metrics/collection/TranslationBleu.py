import sacrebleu
import torch
from sacremoses import MosesTokenizer
from easynmt import EasyNMT

from metrics.collection.MetricClass import MetricClass


class TranslationBleu(MetricClass):
    '''Reference-free BLEU. A nmt system is used to translate the source. Then BLEU is applied. Here we use
    SacreBleu SentenceBleu by
    Matt Post. “A Call for Clarity in Reporting BLEU Scores”. In: Proceedings of the Third Conference on
    Machine Translation: Research Papers. Brussels, Belgium: Association for Computational Linguistics, Oct.
    2018, pp. 186–191. doi: 10.18653/v1/W18-6319. url: https://aclanthology.org/W18-
    6319.
    and as NMT system we use m2m100 by:
    Angela Fan, Shruti Bhosale, Holger Schwenk, Zhiyi Ma, Ahmed El-Kishky, Siddharth Goyal, Man-
    deep Baines, Onur Celebi, Guillaume Wenzek, Vishrav Chaudhary, Naman Goyal, Tom Birch, Vitaliy
    Liptchinsky, Sergey Edunov, Edouard Grave, Michael Auli, and Armand Joulin. Beyond English-Centric
    Multilingual Machine Translation. 2020. arXiv: 2010.11125 [cs.CL].
    To apply it, we use the easynmt library by Nils Reimers (https://github.com/UKPLab/EasyNMT)'''
    ref_based=False
    name = 'TRANSLATIONBLEU'


    def __init__(self):
        self.tokenizer = MosesTokenizer(lang='en')
        self.model = EasyNMT('m2m_100_1.2B')
        self.verbose=True
        self.bs = 4

    def __del__(self):
        # Just deleting this wrapper will not free gpu
        del self.model.translator.model
        del self.model
        torch.cuda.empty_cache()

    def __call__(self, src, hyp, lp = 'de-en', preprocess=False, pre_translations = None):
        '''
        :param ref: A list of strings with reference sentences
        :param hyp: A list of strings with hypothesis sentences
        :return: A list of SacreBleu sentencebleu scores per source - hypothesis pair
        '''
        if preprocess:
            # when no abstraction is used, the translations of the source need to be computed here
            if not pre_translations:
                src_hyp_dict = self.precompute_translations(src, lp)
            else:
                src_hyp_dict = pre_translations
            src = [src_hyp_dict[s] for s in src]

        # lowercasing to strengthen the chances of a word match
        hyp_tok = [self.tokenizer.tokenize(mt, return_str=True).lower() for mt in hyp]
        ref_tok = [self.tokenizer.tokenize(src, return_str=True).lower() for src in src]

        return [sacrebleu.sentence_bleu(hyp_tok[x], [ref_tok[x]]).score/100 if len(hyp_tok[x]) >0 and len(ref_tok[x]) >0 else 0.0 for x in range(len(hyp_tok))]

    def get_abstraction(self, src, lp='et-en', pre_translations=None):
        '''
        As this function needs a language pair for xmoverscore we am overwriting the base
        :param src: A source to be used with every value
        :param ref: A ref to be used with every value
        :return: A function only depending on a list of references
        '''

        # To Do, use the other xmoverscores as well
        s, h = lp.split('-')
        # Either precompute translations or compute them here during runtime, which might be inefficient
        if not pre_translations:
            src_hyp_dict = self.precompute_translations([src], lp)
        else:
            src_hyp_dict = pre_translations
        src = src_hyp_dict[src]
        return lambda hyp: self.__call__([src] * len(hyp), hyp, lp=lp)

    def precompute_translations(self, src, lp):
        s, h = lp.split('-')
        return {src:hyp for src, hyp in zip(src, self.model.translate(src, source_lang=s, target_lang=h, batch_size=self.bs, show_progress_bar=True))}




if __name__ == '__main__':
    b = TranslationBleu()

    # Sample using ref and hyp lists
    print(b(["Ein einfacher Satz als Test"],["A simple sentence for test"], lp='de-en'))
    #[0.10682175159905848]

    # Sample using a fixed reference for a list of hypothesis
    b_trimmed = b.get_abstraction("Ein einfacher Satz als Test")
    print(b_trimmed(["A simple sentence for test", "Another simple sentence for test", 'A test sentence for']))
    #[0.193576934939088, 0.10400597689005303, 0.1937692912686648]
