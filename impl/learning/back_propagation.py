from impl.learning.error_functions.rms import RMS
from impl.learning.least_mean_square import LeastMeanSquare

__author__ = 'Douglas Eric Fonseca Rodrigues'


class BackPropagation(LeastMeanSquare):
    def __init__(self, neural_network, error_function=RMS(), learning_rate=0.1, max_error=0.01, max_iterations=None):
        """
        :type neural_network: core.neural_network.NeuralNetwork
        :type error_function: core.learning.error_functions.error_function.ErrorFunction
        :type learning_rate: float
        :type max_error: float
        :type max_iterations: int
        """
        LeastMeanSquare.__init__(self, neural_network, error_function, learning_rate, max_error, max_iterations)

    def _update_network_weights(self, output_error, i=1):
        """
        :type output_error: list[float]
        :type i: int
        """

        layer = self.neural_network.layers[i]

        if i == len(self.neural_network.layers) - 1:
            # Output layer

            for neuron, error in zip(self.neural_network.output_neurons, output_error):
                delta = error * neuron.activation_function.calculate_derivative(neuron.input)
                LeastMeanSquare._update_neuron_weights(self, neuron, delta)

            return zip(layer.neurons, output_error)
        else:
            next_layer = self._update_network_weights(output_error, i + 1)

            neurons_errors = []

            for neuron in layer.neurons:
                delta_sum = 0.0

                for next_neuron, next_neuron_error in next_layer:
                    if neuron in next_neuron.input_connections:
                        delta_sum += next_neuron.input_connections[neuron].weight * next_neuron_error

                neuron_error = delta_sum * neuron.activation_function.calculate_derivative(neuron.input)
                neurons_errors.append(neuron_error)

                LeastMeanSquare._update_neuron_weights(self, neuron, neuron_error)

            return zip(layer.neurons, neurons_errors)

