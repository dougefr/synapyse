from synapyse.impl.learning.unsupervised_hebbian_learning import UnsupervisedHebbianLearning

__author__ = 'Douglas Eric Fonseca Rodrigues'


class BinaryHebbianLearning(UnsupervisedHebbianLearning):
    def __init__(self, neural_network):
        """
        :type neural_network: core.neural_network.NeuralNetwork
        """
        UnsupervisedHebbianLearning.__init__(self, neural_network, 1)

    def calculate_new_weight(self, connection, input_value, output_value):
        """
        :type connection: core.connection.Connection
        :type input_value: float
        :type output_value: float
        """
        if input_value > 0 and output_value > 0:
            connection.weight += self.learning_rate
        elif input_value <= 0 and output_value <= 0:
            connection.weight += self.learning_rate
        else:
            connection.weight -= self.learning_rate