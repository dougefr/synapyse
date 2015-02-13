from synapyse.impl.learning.back_propagation import BackPropagation

__author__ = 'Douglas Eric Fonseca Rodrigues'


class MomentumBackPropagation(BackPropagation):
    def __init__(self, neural_network, learning_rate, momentum, max_error, max_iterations=None):
        """
        :type neural_network: core.neural_network.NeuralNetwork
        :type learning_rate: float
        :type momentum: float
        :type max_error: float
        :type max_iterations: int
        """
        BackPropagation.__init__(self, neural_network, learning_rate, max_error, max_iterations)

        self.momentum = momentum

        self.previous_weight_values = {}
        """:type : dict[core.connection.Connection, float]"""

    def update_neuron_weights(self, neuron, error):
        """
        :type neuron: core.neuron.Neuron
        :type error: float
        """
        for connection in neuron.input_connections.values():

            connection_input = connection.origin.output

            if connection_input != 0:

                if connection not in self.previous_weight_values:
                    self.previous_weight_values[connection] = 0

                weight_change = self.learning_rate * error * connection_input + self.momentum * (
                    connection.weight - self.previous_weight_values[connection])

                self.previous_weight_values[connection] = connection.weight

                connection.weight += weight_change