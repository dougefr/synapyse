from core.layer import Layer
from core.neural_network import NeuralNetwork
from impl.activation_functions.linear import Linear
from impl.bias_neuron import BiasNeuron

__author__ = 'Douglas Eric Fonseca Rodrigues'


class MultiLayerPerceptron(NeuralNetwork):
    def __init__(self):
        NeuralNetwork.__init__(self)

    def create_layer(self, neuron_count, input_function, activation_function=Linear()):
        """
        :type neuron_count: int
        :type input_function: core.input_functions.input_function.InputFunction
        :type activation_function: core.activation_functions.activation_function.ActivationFunction
        """
        new_layer = Layer(neuron_count, input_function, activation_function)

        if len(self.layers) > 0:
            self.layers[-1].neurons.append(BiasNeuron(activation_function))

        NeuralNetwork.add_layer(self, new_layer)
