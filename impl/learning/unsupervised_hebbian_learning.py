from core.learning.unsupervised_learning import UnsupervisedLearning

__author__ = "Douglas Eric Fonseca Rodrigues"


class UnsupervisedHebbianLearning(UnsupervisedLearning):
    def __init__(self, neural_network, learning_rate):
        """
        :type neural_network: core.neural_network.NeuralNetwork
        :type learning_rate: float
        """
        UnsupervisedLearning.__init__(self, neural_network, learning_rate, 1)

    def update_network_weights(self):
        for neuron in self.neural_network.output_neurons:
            for connection in neuron.input_connections.values():
                self.calculate_new_weight(connection, connection.origin.output, neuron.output)

    def calculate_new_weight(self, connection, input_value, output_value):
        """
        :type connection: core.connection.Connection
        :type input_value: float
        :type output_value: float
        """
        connection.weight += input_value * output_value * self.learning_rate