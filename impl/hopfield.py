from base.neural_network import NeuralNetwork
from impl.hopfield_layer import HopfieldLayer

__author__ = 'Douglas Eric Fonseca Rodrigues'


class Hopfield(NeuralNetwork):
    def __init__(self, neuron_count, y_high, y_low):
        """
        :type neuron_count: int
        :type y_high: float
        :type y_low: float
        """
        NeuralNetwork.__init__(self)
        self.add_layer(HopfieldLayer(neuron_count, y_high, y_low))