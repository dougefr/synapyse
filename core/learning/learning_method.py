from abc import abstractmethod, ABCMeta

__author__ = 'Douglas'


class LearningMethod:
    __metaclass__ = ABCMeta

    def __init__(self, neural_network):
        """
        :type neural_network: NeuralNetwork
        """

        self.neural_network = neural_network
        """:type : NeuralNetwork"""

    @abstractmethod
    def has_reached_stop_condition(self):
        pass