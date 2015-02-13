from abc import ABCMeta, abstractmethod

from base.learning.iterative_learning import IterativeLearning


__author__ = 'Douglas Eric Fonseca Rodrigues'


class UnsupervisedLearning(IterativeLearning):
    __metaclass__ = ABCMeta

    def __init__(self, neural_network, learning_rate, max_iterations=None):
        """
        :type neural_network: core.neural_network.NeuralNetwork
        :type learning_rate: float
        :type max_iterations: int
        """
        IterativeLearning.__init__(self, neural_network, learning_rate, max_iterations)

    def iteration(self, training_set):
        """
        :type training_set: core.learning.training_set.TrainingSet
        """
        for training_set_row in training_set:
            self.neural_network.set_input(training_set_row.input_pattern) \
                .compute()
            self.update_network_weights()

    @abstractmethod
    def update_network_weights(self):
        pass