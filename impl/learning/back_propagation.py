from core.learning.supervised_learning import SupervisedLearning

__author__ = 'Douglas'


class BackPropagation(SupervisedLearning):
    def __init__(self, neural_network, error_function, learning_rate=0.1, momentum=0.0, max_error=0.01, max_iterations=None):

        SupervisedLearning.__init__(self, neural_network, error_function, learning_rate, max_error, max_iterations)

        self.momentum = momentum
        """:type : float"""

    def _update_network_weights(self, output_error, i=0):
        """
        :type output_error: list[float]
        :type i: int
        """

        layer = self.neural_network.layers[i]

        if layer.previous is None:
            # Input layer... nothing to do
            self._update_network_weights(output_error, i + 1)
            return None
        elif i == len(self.neural_network.layers) - 1:
            # Output layer

            for neuron, error in zip(self.neural_network.output_neurons, output_error):
                delta = error * neuron.activation_function.calculate_derivative(neuron.input)
                self.__update_neuron_weights(neuron, delta)

            return zip(layer.neurons, output_error)
        else:
            next_layer = self._update_network_weights(output_error, i + 1)

            neurons_errors = []

            for neuron in layer.neurons:
                delta_sum = 0

                for next_neuron, next_output_error in next_layer:
                    delta_sum += next_neuron.input_connections[neuron].weight * next_output_error

                neuron_error = delta_sum * neuron.activation_function.calculate_derivative(neuron.input)
                neurons_errors.append(neuron_error)

                self.__update_neuron_weights(neuron, neuron_error)

            return zip(layer.neurons, neurons_errors)

    def __update_neuron_weights(self, neuron, error):
        """
        :type neuron: Neuron
        :type error: float
        """
        for connection in neuron.input_connections.values():
            connection.weight += (connection.destination.output * error * self.learning_rate + (self.momentum * connection.weight))