import abc


class Explainer(metaclass=abc.ABCMeta):
    '''
    This class is an abstract class for metrics
    '''

    def explain(self, hyp, metric):
        '''
        This function explains a given metric for an input hypothesis
        Not every Explainer necessarily needs this function, as some of them implement other loops
        :return: Explanation of any dataformat
        '''
        raise NotImplementedError

    @abc.abstractmethod
    def apply_explanation(self, df, metrics = None):
        '''
        This function calls the apply loop of the metric wrapper class to apply the explain method to all data samples
        :param df: A dataframe with SRC, HYP, REF, that should be explained
        :param metrics: The names of the metrics that should  be explained
        :returns: A list of dictionaries with results. One dictionary per row. In case of feature importance, each list element should
        be structured as follows:
        {"src": "S\u00f5da ja majanduslik surve s\u00fcvendasid juba olemasolevaid killustumise protsesse ja klassikonflikte sotsiaalsete klasside vahel ja sees.",
        "ref": "War and economic pressure further deepened the existing fragmentation processes and class conflicts between and within social classes.",
        "hyp": "War and economic pressure have deepened the processes of fragmentation and class conflicts that already exist between and within the social classes .",
        "metrics": {
            "BLEU": {"attributions": [[0.02885010939375271, "War"], [0.07879525360059099, "and"], ...], "score": 0.2514807689508543},
            "BERTSCORE": {"attributions": [[0.06219312217500475, "War"], [0.014879471725887723, "and"], ...], "score": 0.9699435234069823}},
            ...
        "times": {
            "BLEU": {"time": 1.36393404006958},
            "BERTSCORE": {"time": 1.3575000762939453}},
        "corpus_row": 0},
            ...]

        It may contain further information. Mandatory are the sentences and the `attributions´ for each metric per sample.
        '''
        return NotImplementedError