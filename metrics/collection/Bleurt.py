import zipfile
import requests

from io import BytesIO
from os import path, environ, makedirs
from bleurt import score as bleurt_score

from metrics.collection.MetricClass import MetricClass


class Bleurt(MetricClass):
    '''
    A wrapper class for BLEURT (https://github.com/google-research/bleurt), by
    Thibault Sellam, Dipanjan Das, and Ankur Parikh. “BLEURT: Learning Robust Metrics for Text Gen-
    eration”. In: Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics.
    Online: Association for Computational Linguistics, July 2020, pp. 7881–7892. doi: 10.18653/v1/2020.acl-main.704.
    url: https://www.aclweb.org/anthology/2020.acl-main.704.
    '''
    ref_based = True
    name = 'BLEURT'


    def __init__(self, model_path=None, model_name="BLEURT-base-128", bs=32):
        '''
        :param model_path: A path to an existing BLEURT model. If None, it will download the model and cache it on a
                           location based on the HOME environment variable.
        :param model_name: The BLEURT model to use
        :param bs: The batch size
        '''
        if model_path:
            self.model_path = path.join(model_path, model_name)
        else:
            self.model_path = path.join(self.get_cache_folder(), model_name)
        self.model_url = "https://storage.googleapis.com/bleurt-oss/" + model_name + ".zip"

        if not path.exists(self.model_path):
            print("Downloading bleurt checkpoint. This might take a while. Downloading to:", self.model_path)
            self.download_model()

        self.scorer = bleurt_score.BleurtScorer(path.join(self.model_path, model_name))
        self.bs = bs

    def get_cache_folder(self):
        if "HOME" in environ:
            cache_directory = path.join(environ["HOME"], "\\.cache\\bleurt")
            if not path.exists(cache_directory):
                makedirs(cache_directory)
            return cache_directory
        else:
            raise Exception("HOME environment variable is not defined.")

    def download_model(self):
        with zipfile.ZipFile(BytesIO(requests.get(self.model_url.lower()).content)) as zipmodel:
            zipmodel.extractall(self.model_path)

    def __call__(self, ref, hyp):
        '''
        :param ref: A list of strings with reference sentences
        :param hyp: A list of strings with hypothesis sentences
        :return: A list of Bleurt Scores
        '''

        return self.scorer.score(ref, hyp, batch_size=self.bs)


if __name__ == '__main__':
    # We are having some issues with the usage of Tensorflow here. Using this solution from
    # https://stackoverflow.com/questions/41117740/tensorflow-crashes-with-cublas-status-alloc-failed
    #  An alternative can be to run the tensorflow part in a subprocess, which would be kind of ugly
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
    ##################################################################################################

    b = Bleurt()

    # Sample using ref and hyp lists
    print(b(["A test sentence"], ["A simple sentence for test"]))
    # [-0.29165118932724]


    # Sample using a fixed reference for a list of hypothesis
    b_trimmed = b.get_abstraction("A test sentence")
    print(b_trimmed(["A simple sentence for test", "Another simple sentence for test", 'A test sentence']))
    # [-0.29165172576904297, -0.44685500860214233, 1.0369492769241333]
