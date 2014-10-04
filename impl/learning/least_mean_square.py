from core.learning.supervised_learning import SupervisedLearning
from impl.learning.error_functions.rms import RMS

__author__ = 'Douglas Eric Fonseca Rodrigues'


class LeastMeanSquare(SupervisedLearning):
    def __init__(self, neural_network, error_function=RMS(), learning_rate=0.1, max_error=0.01, max_iterations=None):
        """
        :type neural_network: core.neural_network.NeuralNetwork
        :type error_function: core.learning.error_functions.error_function.ErrorFunction
        :type learning_rate: float
        :type max_error: float
        :type max_iterations: int
        """
        SupervisedLearning.__init__(self, neural_network, error_function, learning_rate, max_error, max_iterations)

    def update_network_weights(self, output_error):
        for neuron, error in zip(self.neural_network.output_neurons, output_error):
            self.update_neuron_weights(neuron, error)

    def update_neuron_weights(self, neuron, error):
        """
        :type neuron: core.neuron.Neuron
        :type error: float
        """
        for connection in neuron.input_connections.values():
            connection.weight += connection.origin.output * error * self.learning_rate
