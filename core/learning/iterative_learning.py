from abc import abstractmethod, ABCMeta

from core.learning.learning_method import LearningMethod


__author__ = 'Douglas Eric Fonseca Rodrigues'


class IterativeLearning(LearningMethod):
    __metaclass__ = ABCMeta

    def __init__(self, neural_network, learning_rate=0.1, max_iterations=None):
        """
        :type neural_network: core.neural_network.NeuralNetwork
        :type learning_rate: float
        :type max_iterations: int
        """

        LearningMethod.__init__(self, neural_network)

        self.max_iterations = max_iterations
        self.actual_iteration = 0
        self.learning_rate = learning_rate

    def learn(self, training_set):
        """
        :type training_set: core.learning.training_set.TrainingSet
        """
        while not self.has_reached_stop_condition():
            self.actual_iteration += 1
            self.iteration(training_set)

    def has_reached_stop_condition(self):
        return self.max_iterations == self.actual_iteration

    @abstractmethod
    def iteration(self, training_set):
        """
        :type training_set: core.learning.training_set.TrainingSet
        """
        pass