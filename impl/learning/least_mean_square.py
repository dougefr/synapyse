from core.learning.supervised_learning import SupervisedLearning

__author__ = 'douglas'


class LeastMeanSquare(SupervisedLearning):
    def __init__(self, neural_network, error_function, learning_rate=0.1, max_error=0.01, max_iterations=None):
        SupervisedLearning.__init__(self, neural_network, error_function, learning_rate, max_error, max_iterations)

    def _update_network_weights(self, output_error):
        for neuron, error in zip(self.neural_network.output_neurons, output_error):
            self._update_neuron_weights(neuron, error)

    def _update_neuron_weights(self, neuron, error):
        """
        :type neuron: Neuron
        :type error: float
        """
        for connection in neuron.input_connections.values():
            connection.weight += connection.origin.output * error * self.learning_rate
