import abc


class MetricClass(metaclass=abc.ABCMeta):
    '''
    This class is an abstract class for metrics
    '''

    @abc.abstractmethod
    def __call__(self, ref, hyp):
        '''
        This function calculates a metric given all of its parameters (ref could also be src)
        :return: score
        '''
        raise NotImplementedError

    def get_abstraction(self, ref):
        '''
        The input to this function are parameters that should be set for the function
        :param ref: A string that should be used as reference for all hypothesis (could also use src instead)
        :return: A metric only depending on one parameter - hyp
        '''
        return lambda hyp: self.__call__([ref] * len(hyp), hyp)
