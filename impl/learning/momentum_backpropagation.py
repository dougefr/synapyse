from impl.learning.back_propagation import BackPropagation
from impl.learning.error_functions.rms import RMS

__author__ = 'Douglas Eric Fonseca Rodrigues'


class MomentumBackpropagation(BackPropagation):
    def __init__(self, neural_network, momentum, learning_rate, max_error, max_iterations=None,
                 error_function=RMS()):
        """
        :type neural_network: core.neural_network.NeuralNetwork
        :type momentum: float
        :type learning_rate: float
        :type max_error: float
        :type max_iterations: int
        :type error_function: core.learning.error_functions.error_function.ErrorFunction
        """
        BackPropagation.__init__(self, neural_network, learning_rate, max_error, max_iterations, error_function)

        self.momentum = momentum

        self.previous_weight_values = {}
        """:type : dict[core.connection.Connection, float]"""

    def update_neuron_weights(self, neuron, error):
        """
        :type neuron: core.neuron.Neuron
        :type error: float
        """
        for connection in neuron.input_connections.values():
            if connection.input != 0:

                if connection not in self.previous_weight_values:
                    self.previous_weight_values[connection] = 0

                weight_change = self.learning_rate * error * connection.input + self.momentum * (
                    connection.weight - self.previous_weight_values[connection])

                self.previous_weight_values[connection] = connection.weight

                connection.weight += weight_change