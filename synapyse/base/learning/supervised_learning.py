from abc import ABCMeta, abstractmethod
import logging

from synapyse.base.learning.iterative_learning import IterativeLearning
from synapyse.impl.learning.error_functions.rms import RMS


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

        logger = logging.getLogger('synapyse')

        for training_set_row in training_set:

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug('SupervisedLearning::iteration: processing input_pattern=' +
                             str(training_set_row.input_pattern) +
                             ' ideal_output=' +
                             str(training_set_row.ideal_output))

            computed_output = self.neural_network.set_input(training_set_row.input_pattern) \
                .compute() \
                .output

            # Calculate the output error
            output_error = [ideal - actual for ideal, actual in zip(training_set_row.ideal_output, computed_output)]
            error_function.add_error(output_error)

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug('SupervisedLearning::iteration: computed_output=' + str(computed_output))
                logger.debug('SupervisedLearning::iteration: output_error=' + str(output_error))

            self.update_network_weights(output_error)

        self.total_network_error = error_function.total_error

    def has_reached_stop_condition(self):
        return IterativeLearning.has_reached_stop_condition(self) or (
            self.total_network_error is not None and self.total_network_error <= self.max_error)

    @abstractmethod
    def update_network_weights(self, output_error):
        """
        :type output_error: list[float]
        """
        pass