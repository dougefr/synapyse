from core.neural_network import NeuralNetwork
from impl.hopfield_layer import HopfieldLayer

__author__ = 'Douglas Eric Fonseca Rodrigues'


class Hopfield(NeuralNetwork):
    def __init__(self, neuron_count):
        """
        :type neuron_count: int
        """
        NeuralNetwork.__init__(self)
        self.add_layer(HopfieldLayer(neuron_count))