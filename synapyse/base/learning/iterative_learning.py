from abc import abstractmethod, ABCMeta

from synapyse.base.learning.learning_method import LearningMethod
from synapyse.util.logger import Logger


__author__ = 'Douglas Eric Fonseca Rodrigues'


class IterativeLearning(LearningMethod):
    __metaclass__ = ABCMeta

    def __init__(self, neural_network, learning_rate, max_iterations=None):
        """
        :type neural_network: synapyse.base.neural_network.NeuralNetwork
        :type learning_rate: float
        :type max_iterations: int
        """
        LearningMethod.__init__(self, neural_network)

        self.max_iterations = max_iterations
        self.actual_iteration = 0
        self.learning_rate = learning_rate

        self.on_before_iteration = lambda iterative_learning: None
        self.on_after_iteration = lambda iterative_learning: None

    def learn(self, training_set):
        """
        :type training_set: synapyse.base.learning.training_set.TrainingSet
        """
        while not self.has_reached_stop_condition():
            self.actual_iteration += 1

            self.on_before_iteration(self)
            self.iteration(training_set)
            self.on_after_iteration(self)

            if Logger.is_debug_enabled():
                Logger.debug('IterativeLearning::learn: actual_iteration=', self.actual_iteration)

    def has_reached_stop_condition(self):
        return self.max_iterations == self.actual_iteration

    @abstractmethod
    def iteration(self, training_set):
        """
        :type training_set: synapyse.base.learning.training_set.TrainingSet
        """
        pass