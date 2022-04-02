import torch
from comet import download_model, load_from_checkpoint

from metrics.collection.MetricClass import MetricClass


class Comet(MetricClass):
    '''
    A wrapper class for COMET (https://github.com/Unbabel/COMET) by:
    Ricardo Rei, Craig Stewart, Ana C Farinha, and Alon Lavie. “COMET: A Neural Framework for
    MT Evaluation”. In: Proceedings of the 2020 Conference on Empirical Methods in Natural Language
    Processing (EMNLP). Online: Association for Computational Linguistics, Nov. 2020, pp. 2685–2702. doi:
    10.18653/v1/2020.emnlp-main.213. url: https://aclanthology.org/2020.emnlp-main.213.
    '''

    # To install comet on windows follow the workaround here: https://github.com/Unbabel/COMET/issues/17
    # i.e. you need to temporarily install torch 1.6.0 during setup. Afterwards you may install a newer one
    # Additionally on windows, there is a change in fairseqs multilingual_masked_lm.py line 73, as it will split the path at : which
    # we don't want in case of windows paths --> dictionary = Dictionary.load(os.path.join(args.data, 'dict.txt'))

    # UPDATE: the newer version of unbabel requires torch 1.6 instead : pip install unbabel-comet==1.0.0rc4
    # In Windows you need to set num_workers=0 for all Dataloaders of comet to prevent it to try to pickle lambda
    # An alternative would be to make the object writable
    ref_based = True

    def __init__(self, verbose=True, model= "wmt20-comet-da", model_path=None):
    #def __init__(self, verbose=True, model="wmt-large-da-estimator-1718", model_path=None):
        model_path = download_model(model)
        self.model = load_from_checkpoint(model_path)
        self.verbose = verbose
        self.bs = 16

    def __del__(self):
        # Just deleting this wrapper will not free gpu
        del self.model
        torch.cuda.empty_cache()

    def __call__(self, src, ref, hyp):
        '''
        :param src: A list of strings with source sentences
        :param ref: A list of strings with reference sentences
        :param hyp: A list of strings with hypothesis sentences
        :return: A list of comet scores
        '''
        data = [{"src": s, "mt": h, "ref": r} for s, h, r in zip(src, hyp, ref)]
        results, _ = self.model.predict(data, batch_size=self.bs, gpus=1)

        return results

    def get_abstraction(self, src, ref):
        '''
        As this function needs two inputs for comet we are overwriting the base
        :param src: A source to be used with every value
        :param ref: A ref to be used with every value
        :return: A function only depending on a list of hypotheses
        '''

        return lambda hyp: self.__call__([src] * len(hyp), [ref] * len(hyp), hyp)


if __name__ == '__main__':
    c = Comet()
    name = 'COMET'


    # Sample using src, ref, hyp lists
    print(c(["Ein Test Satz"], ["A test sentence"], ["A simple sentence for test"]))
    # [0.5879985094070435]

    # Sample using a fixed source and reference for a list of hypothesis
    c_trimmed = c.get_abstraction("Ein Test Satz", "A test sentence")
    print(c_trimmed(["A simple sentence for test", "Another simple sentence for test", 'A test sentence']))
    # [0.5877664685249329, 0.5560164451599121, 1.2612366676330566]