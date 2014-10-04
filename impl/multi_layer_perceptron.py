from core.layer import Layer
from core.neural_network import NeuralNetwork
from impl.activation_functions.linear import Linear
from impl.bias_neuron import BiasNeuron
from impl.learning.back_propagation import BackPropagation
from impl.learning.error_functions.rms import RMS

__author__ = 'Douglas'


class MultiLayerPerceptron(NeuralNetwork):
    def __init__(self, learning_rate=0.1, max_error=0.01, max_iterations=None):
        NeuralNetwork.__init__(self, BackPropagation(self, RMS(), learning_rate, max_error, max_iterations))

    def add_layer(self, neuron_count, input_function, activation_function=Linear()):
        """
        :type neuron_count: int
        :type input_function: InputFunction
        :type activation_function:
        """
        new_layer = Layer(neuron_count, input_function, activation_function)

        if len(self.layers) > 0:
            self.layers[-1].neurons.append(BiasNeuron(activation_function))

        NeuralNetwork._add_layer(self, new_layer)
