import logging
from synapyse.base.learning.supervised_learning import SupervisedLearning

__author__ = 'Douglas Eric Fonseca Rodrigues'


class LeastMeanSquare(SupervisedLearning):
    def __init__(self, neural_network, learning_rate, max_error, max_iterations=None):
        """
        :type neural_network: core.neural_network.NeuralNetwork
        :type learning_rate: float
        :type max_error: float
        :type max_iterations: int
        """
        SupervisedLearning.__init__(self, neural_network, learning_rate, max_error, max_iterations)

    def update_network_weights(self, output_error):
        for neuron, error in zip(self.neural_network.output_neurons, output_error):
            self.update_neuron_weights(neuron, error)

    def update_neuron_weights(self, neuron, error):
        """
        :type neuron: core.neuron.Neuron
        :type error: float
        """
        logger = logging.getLogger('synapyse')

        for connection in neuron.input_connections.values():

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug('LeastMeanSquare::update_neuron_weights: weight before =' + str(connection.weight))

            connection.weight += connection.origin.output * error * self.learning_rate

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug('LeastMeanSquare::update_neuron_weights: weight after =' + str(connection.weight))
