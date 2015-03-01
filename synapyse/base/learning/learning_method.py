from abc import abstractmethod, ABCMeta

__author__ = 'Douglas Eric Fonseca Rodrigues'


class LearningMethod:
    __metaclass__ = ABCMeta

    def __init__(self, neural_network):
        """
        :type neural_network: synapyse.base.neural_network.NeuralNetwork
        """
        self.neural_network = neural_network

    @abstractmethod
    def has_reached_stop_condition(self):
        pass

    @abstractmethod
    def learn(self, training_set):
        """
        :type training_set: synapyse.base.learning.training_set.TrainingSet
        """
        pass