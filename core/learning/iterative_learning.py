from abc import abstractmethod, ABCMeta

from core.learning.learning_method import LearningMethod


__author__ = 'Douglas'


class IterativeLearning(LearningMethod):
    __metaclass__ = ABCMeta

    def __init__(self, neural_network, learning_rate=0.1, max_iterations=None):
        """
        :type max_iterations: int
        :type neural_network: NeuralNetwork
        """

        LearningMethod.__init__(self, neural_network)

        self.max_iterations = max_iterations
        """:type : int"""

        self.actual_iteration = 0
        """:type : int"""

        self.learning_rate = learning_rate
        """:type : float"""

    def learn(self, training_set):
        """
        :type training_set: TrainingSet
        """
        while not self.has_reached_stop_condition():
            self.actual_iteration += 1
            self._iteration(training_set)


    def has_reached_stop_condition(self):
        return self.max_iterations == self.actual_iteration

    @abstractmethod
    def _iteration(self, training_set):
        """
        :type training_set: TrainingSet
        """
        pass