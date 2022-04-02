import os
import time

import nltk
import torch
from sacremoses import MosesTokenizer
from easynmt import EasyNMT
from tqdm import tqdm
import json

from metrics.collection.MetricClass import MetricClass
from project_root import ROOT_DIR


class TranslationMeteor(MetricClass):
    '''Reference-free Meteor. A nmt system is used to translate the source. Then Meteor is applied. Here we use
        the NLTK Meteor (https://www.nltk.org/_modules/nltk/translate/meteor_score.html). Meteor is a metric by:
        Alon Lavie, Kenji Sagae, and Shyamsundar Jayaraman. “The Significance of Recall in Automatic
        Metrics for MT Evaluation”. In: Machine Translation: From Real Users to Research. Ed. by Robert E.
        Frederking and Kathryn B. Taylor. Berlin, Heidelberg: Springer Berlin Heidelberg, 2004, pp. 134–143.
        isbn: 978-3-540-30194-3.
        and as NMT system we use m2m100 by:
        Angela Fan, Shruti Bhosale, Holger Schwenk, Zhiyi Ma, Ahmed El-Kishky, Siddharth Goyal, Man-
        deep Baines, Onur Celebi, Guillaume Wenzek, Vishrav Chaudhary, Naman Goyal, Tom Birch, Vitaliy
        Liptchinsky, Sergey Edunov, Edouard Grave, Michael Auli, and Armand Joulin. Beyond English-Centric
        Multilingual Machine Translation. 2020. arXiv: 2010.11125 [cs.CL].
        To apply it, we use the easynmt library by Nils Reimers (https://github.com/UKPLab/EasyNMT)'''
    ref_based=False
    name = 'TRANSLATIONMETEOR'


    def __init__(self):
        self.tokenizer = MosesTokenizer(lang='en')
        self.model = EasyNMT('m2m_100_1.2B')
        self.bs = 8

        # If this is true, the class methods will require a word by word dictionary to translate
        self.use_word_by_word=False
        self.wbw = None
        if self.use_word_by_word:
            with open(os.path.join(ROOT_DIR,'metrics/corpora/google_word_by_word_dicts/de_zh_dict_eval4nlp_test.json')) as json_file:
                self.wbw = json.load(json_file)
                self.wbw = {k.lower(): v for k, v in self.wbw.items()}

    def __del__(self):
        # Just deleting this wrapper will not free gpu
        del self.model.translator.model
        del self.model
        torch.cuda.empty_cache()

    def __call__(self, src, hyp, lp = 'de-en', preprocess=False, pre_translations=None):
        '''
        :param ref: A list of strings with reference sentences
        :param hyp: A list of strings with hypothesis sentences
        '''
        if preprocess:
            if not self.use_word_by_word:
                if not pre_translations:
                    src_hyp_dict = self.precompute_translations(src, lp)
                else:
                    src_hyp_dict = pre_translations
                src = [src_hyp_dict[s] for s in src]
            else:
                assert self.wbw != None
                src = [' '.join([self.wbw[w.lower()] for w in s.split(' ')]) for s in tqdm(src, desc='preprocessed translations:')]

        hyp = [self.tokenizer.tokenize(mt, return_str=True).lower() for mt in hyp]
        src = [self.tokenizer.tokenize(src, return_str=True).lower() for src in src]
        return [nltk.translate.meteor_score.meteor_score([r], h) for r, h
         in zip(src, hyp)]

    def get_abstraction(self, src, lp='et-en', pre_translations = None):
        # To Do, use the other xmoverscores as well
        s, h = lp.split('-')
        if not self.use_word_by_word:
            if not pre_translations:
                src_hyp_dict = self.precompute_translations([src], lp)
            else:
                src_hyp_dict = pre_translations
            src = src_hyp_dict[src]

        else:
            assert self.wbw != None
            src = ' '.join([self.wbw[w.lower()] for w in src.split(' ')])
        return lambda hyp: self.__call__([src] * len(hyp), hyp, lp=lp)

    def word_by_word(self, src, lp):
        s, h = lp.split('-')
        res = self.model.translate(src, source_lang=s, target_lang=h)
        return res

    def build_word_by_word_dict(self, src_sents, lp, prec = None):
        s, h = lp.split('-')
        src_words = [s.split(' ') for s in src_sents]
        src_words = list(set([item for sublist in src_words for item in sublist]))

        # Use a local model for word lookup
        # pseudo_ref = self.model.translate(src_words, source_lang=s, target_lang=h)

        # Use google translate for word lookup
        from googletrans import Translator
        translator = Translator()
        pseudo_ref = {}
        if prec:
            with open(prec) as json_file:
                pseudo_ref = json.load(json_file)

        # Retry up to ten times once it fails
        for src_word in tqdm(src_words):
            for attempt in range(10):
                print("Attempt:", attempt)
                try:
                    if not src_word in pseudo_ref:
                        if h == 'zh':
                            h = 'zh-CN'
                        pseudo_ref[src_word] = translator.translate(src_word, src=s, dest=h).text
                except Exception as e:
                    print(e)
                    time.sleep(70)

                    if attempt == 9:
                        pseudo_ref[src_word] = 'dummy'
                    continue
                break

        # res = self.model.translate(src, source_lang=s,target_lang=h)
        print(pseudo_ref)
        return pseudo_ref

    def precompute_translations(self, src, lp):
        if self.use_word_by_word:
            return None
        s, h = lp.split('-')
        return {src:hyp for src, hyp in zip(src, self.model.translate(src, source_lang=s, target_lang=h, batch_size=self.bs, show_progress_bar=True))}



if __name__ == '__main__':
    b = TranslationMeteor()

    '''
    This block can build wbw dicts
    '''
    import os
    from project_root import ROOT_DIR
    ro_en_path = os.path.join(ROOT_DIR,'metrics/corpora/pandas_corpora/eval4NLP/eval4nlp_test_de-zh.tsv')
    import pandas as pd

    seg_df = pd.read_csv(ro_en_path, delimiter='\t')
    src_list = seg_df['SRC'].tolist()
    lp = seg_df['LP'].tolist()[0]
    dictionary = b.build_word_by_word_dict(src_list, lp,None)

    save_path = os.path.join(ROOT_DIR,'metrics/corpora/google_word_by_word_dicts/de_zh_dict_eval4nlp_test.json')
    with open(save_path, 'w') as json_file:
        json.dump(dictionary,json_file)


    # Sample using ref and hyp lists
    print(b(["Ein einfacher Satz als Test"],["A simple sentence for test"]))
    # [0.10000000000000002]

    # Sample using a fixed reference for a list of hypothesis
    b_trimmed = b.get_abstraction("Ein einfacher Satz als Test")
    print(b_trimmed(["A simple sentence for test", "Another simple sentence for test", 'A test sentence for']))
    # [0.25423728813559315, 0.1694915254237288, 0.32327586206896547]