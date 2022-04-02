import nltk

from metrics.collection.MetricClass import MetricClass


class Meteor(MetricClass):
    '''A wrapper for NLTK meteor (https://www.nltk.org/_modules/nltk/translate/meteor_score.html). Meteor is a metric by:
    Alon Lavie, Kenji Sagae, and Shyamsundar Jayaraman. “The Significance of Recall in Automatic
    Metrics for MT Evaluation”. In: Machine Translation: From Real Users to Research. Ed. by Robert E.
    Frederking and Kathryn B. Taylor. Berlin, Heidelberg: Springer Berlin Heidelberg, 2004, pp. 134–143.
    isbn: 978-3-540-30194-3.'''
    ref_based = True
    name = 'METEOR'


    def __call__(self, ref, hyp):
        '''
        :param ref: A list of strings with reference sentences
        :param hyp: A list of strings with hypothesis sentences
        :return: A list of METEOR scores per reference - hypothesis pair
        '''

        return [nltk.translate.meteor_score.meteor_score([r], h) for r, h
                in zip(ref, hyp)]


if __name__ == '__main__':
    m = Meteor()

    # Sample using ref and hyp lists
    print(m(["A test sentence"], ["A simple sentence for test"]))
    # [0.46875]

    # Sample using a fixed reference for a list of hypothesis
    m_trimmed = m.get_abstraction("A test sentence")
    print(m_trimmed(["A simple sentence for test", "Another simple sentence for test", 'A test sentence']))
    # [0.46875, 0.3125, 0.9814814814814815]
