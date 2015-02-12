from core.neural_network import NeuralNetwork
from impl.bias_neuron import BiasNeuron

__author__ = 'Douglas Eric Fonseca Rodrigues'


class MultiLayerPerceptron(NeuralNetwork):
    def __init__(self):
        NeuralNetwork.__init__(self)

    def add_layer(self, new_layer):
        """
        :type new_layer: core.layer.Layer
        """
        if len(self.layers) > 0:
            self.layers[-1].neurons.append(BiasNeuron(new_layer.activation_function))

        NeuralNetwork.add_layer(self, new_layer)

        return self