import torch

from metrics.collection.MetricClass import MetricClass


class BertScore(MetricClass):
    '''
    A wrapper class for BERTScore from https://github.com/Tiiiger/bert_score by
    Tianyi Zhang, Varsha Kishore, Felix Wu, Kilian Q. Weinberger, and Yoav Artzi. “BERTScore: Evaluating
    Text Generation with BERT”. In: International Conference on Learning Representations. 2020. url:
    https://openreview.net/forum?id=SkeHuCVFDr.
    '''

    # The standard variant of BertScore is reference based. If you use it in the multilingual setup, you should ignore this
    # property in the evaluation loop.
    ref_based = True
    name = 'BERTSCORE'

    def __init__(self, verbose=False, num_layers=17, model='joeddav/xlm-roberta-large-xnli', custom_bert_scorer=False,
                 idf_sents=None):
        '''
        The default configuration runs roberta-large with the 17th layer.
        :param verbose: Whether to print out progress
        :param num_layers: The layer to choose embeddings from. This parameter is only active if custom_bert_scorer = True
        :param model: The model to choose from huggingface model hub. This parameter is only active if custom_bert_scorer = True.
                      New models need to be registered in metrics_libs/bert_score/utils.py
        :param custom_bert_scorer: If true, metrics_libs/bert_score will be used. Otherwise the BERTScore version that is installed via pip.
                                   Visit the respective folder for more information.
        :param idf_sents: Sentences to use for BERTScore's idf weighting. Cuttently this is only settable for custom_bert_scorer. But can be easily
                          added to the standard configuration. If none are specified, no weighting is used.
        '''
        self.custom_bert_scorer = custom_bert_scorer
        if custom_bert_scorer == False:
            from bert_score import BERTScorer
            self.scorer = BERTScorer(lang='en', batch_size=32)
        else:
            from metrics.collection.metrics_libs.bert_score.scorer import BERTScorer
            if idf_sents:
                self.scorer = BERTScorer(model_type=model, batch_size=32, num_layers=num_layers, idf=True,
                                         idf_sents=idf_sents)
            else:
                self.scorer = BERTScorer(model_type=model, batch_size=32, num_layers=num_layers)

        self.verbose = verbose

    def __call__(self, ref, hyp):
        '''
        Implementation from here, installed via pip: https://github.com/Tiiiger/bert_score
        :param ref: A list of strings with reference sentences
        :param hyp: A list of strings with hypothesis sentences
        :return: A list of f1 values of BertScore (if custom_bert_scorer ==False)
                 A list of f1 values, a list of hypothesis importance scores per sentence and a list of source importance scores per sentence
        '''
        if self.custom_bert_scorer == False:
            return self.scorer.score(hyp, ref, verbose=self.verbose)[2].tolist()

        else:
            res = self.scorer.score(hyp, ref, verbose=self.verbose)
            return res[0][2].tolist(), res[1], res[2]

    def __del__(self):
        # Just deleting this wrapper will sometimes not free gpu
        del self.scorer._model
        del self.scorer
        torch.cuda.empty_cache()


if __name__ == '__main__':
    b = BertScore()

    # Sample using ref and hyp lists
    print(b(["A test sentence"], ["A simple sentence for test"]))
    # [0.9033169150352478]

    # Sample using a fixed reference for a list of hypothesis
    b_trimmed = b.get_abstraction("A test sentence")
    print(b_trimmed(["A simple sentence for test", "Another simple sentence for test", 'A test sentence']))
    # [0.9033169746398926, 0.8992820382118225, 0.9999999403953552]


    b = BertScore(custom_bert_scorer=True)
    # Sample using src and hyp lists
    print(b(["Ein Test Satz"], ["A simple sentence for test"]))
    # ([0.8807982802391052],
    # [[(0.9802274107933044, '<s>'), (0.9372762441635132, '▁A'), (0.8102121353149414, '▁simple'), (0.8539739847183228, '▁sentence'), (0.8423895835876465, '▁for'), (0.892814040184021, '▁test'), (0.9994502663612366, '</s>')]],
    # [[(0.9802274107933044, '<s>'), (0.9372762441635132, '▁Ein'), (0.892814040184021, '▁Test'), (0.8539739847183228, '▁Satz'), (0.9994502663612366, '</s>')]])