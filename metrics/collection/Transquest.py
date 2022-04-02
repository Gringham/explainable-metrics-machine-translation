from metrics.collection.MetricClass import MetricClass
from transquest.algo.sentence_level.monotransquest.run_model import MonoTransQuestModel

class Transquest(MetricClass):
    '''
    A wrapper for TransQuest (https://github.com/TharinduDR/TransQuest), a metric by:
    Tharindu Ranasinghe, Constantin Orasan, and Ruslan Mitkov. “TransQuest: Translation Quality
    Estimation with Cross-lingual Transformers”. In: Proceedings of the 28th International Conference on
    Computational Linguistics. Barcelona, Spain (Online): International Committee on Computational
    Linguistics, Dec. 2020, pp. 5070–5081. doi: 10.18653/v1/2020.coling-main.445. url:
    https://aclanthology.org/2020.coling-main.445.
    '''
    ref_based = False
    name = 'TRANSQUEST'


    def __init__(self, verbose=True):
        # Using the scorer here, as it is advised
        # to cache the model in cases of multiple executions
        self.verbose = verbose
        self.src = 'et'
        self.hyp = 'en'
        self.model = MonoTransQuestModel(
                'xlmroberta',
                f'TransQuest/monotransquest-da-{self.src}_{self.hyp}-wiki', num_labels=1, use_cuda=True)

    def __call__(self, src, hyp, lp = 'et-en'):
        s, h = lp.split('-')
        if self.src != s or self.hyp!=h:
            try:
                # fetch model for 2 languages if possible
                self.model = MonoTransQuestModel(
                    'xlmroberta',
                    f'TransQuest/monotransquest-da-{s}_{h}-wiki', num_labels=1, use_cuda=True)
            except:
                if h == 'en':
                    self.model = MonoTransQuestModel(
                        'xlmroberta',
                        f'TransQuest/monotransquest-da-any_{h}', num_labels=1, use_cuda=True)
            self.src = s
            self.hyp = h
        predictions, raw_outputs = self.model.predict(list(map(list, zip(src, hyp))))
        predictions = predictions.tolist()
        if type(predictions) == float:
            return [predictions]
        return predictions

    def get_abstraction(self, src, lp='et-en'):
        '''
        As this function needs a language pair for xmoverscore we are overwriting the base
        :param src: A source to be used with every value
        :param ref: A ref to be used with every value
        :return: A function only depending on a list of references
        '''

        # To Do, use the other xmoverscores as well
        return lambda hyp: self.__call__([src] * len(hyp), hyp, lp=lp)


if __name__ == '__main__':
    b = Transquest()

    # Sample using ref and hyp lists
    print(b(["A test sentence"], ["A simple sentence for test"]))
    #[0.7177734375]

    # Sample using a fixed reference for a list of hypothesis
    b_trimmed = b.get_abstraction("A test sentence")
    print(b_trimmed(["A simple sentence for test", "Another simple sentence for test", 'A test sentence']))
    #[0.7177734375, 0.64404296875, 0.76806640625]