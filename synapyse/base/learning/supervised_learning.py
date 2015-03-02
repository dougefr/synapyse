from abc import ABCMeta, abstractmethod

from synapyse.base.learning.iterative_learning import IterativeLearning
from synapyse.impl.learning.error_functions.rms import RMS
from synapyse.util.logger import Logger


__author__ = 'Douglas Eric Fonseca Rodrigues'


class SupervisedLearning(IterativeLearning):
    __metaclass__ = ABCMeta

    def __init__(self, neural_network, learning_rate, max_error, max_iterations=None):
        """
        :type neural_network: synapyse.base.neural_network.NeuralNetwork
        :type learning_rate: float
        :type max_error: float
        :type max_iterations: int
        """
        IterativeLearning.__init__(self, neural_network, learning_rate, max_iterations)
        self.max_error = max_error
        self.total_network_error = None

    def iteration(self, training_set):
        """
        :type training_set: synapyse.base.learning.training_set.TrainingSet
        """
        error_function = RMS(len(training_set))
        error_function.reset()

        for training_set_row in training_set:

            if Logger.is_debug_enabled():
                Logger.debug('SupervisedLearning::iteration: processing input_pattern=',
                             training_set_row.input_pattern,
                             ' ideal_output=',
                             training_set_row.ideal_output)

            computed_output = self.neural_network.set_input(training_set_row.input_pattern) \
                .compute() \
                .output

            # Calculate the output error
            output_error = [ideal - actual for ideal, actual in zip(training_set_row.ideal_output, computed_output)]
            error_function.add_error(output_error)

            if Logger.is_debug_enabled():
                Logger.debug('SupervisedLearning::iteration: computed_output=', computed_output)
                Logger.debug('SupervisedLearning::iteration: output_error=', output_error)

            self.update_network_weights(output_error)

        self.total_network_error = error_function.total_error

        if Logger.is_debug_enabled():
            Logger.debug('SupervisedLearning::iteration: total_network_error=', self.total_network_error)

    def has_reached_stop_condition(self):
        return IterativeLearning.has_reached_stop_condition(self) or (
            self.total_network_error is not None and self.total_network_error <= self.max_error)

    @abstractmethod
    def update_network_weights(self, output_error):
        """
        :type output_error: list[float]
        """
        pass