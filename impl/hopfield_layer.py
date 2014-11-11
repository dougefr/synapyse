from core.layer import Layer
from impl.activation_functions.step import Step
from impl.input_functions.weighted_sum import WeightedSum

__author__ = 'Douglas Eric Fonseca Rodrigues'


class HopfieldLayer(Layer):
    def __init__(self, neuron_count):
        """
        :type neuron_count: int
        """
        Layer.__init__(self, neuron_count, WeightedSum(), Step())
        self.connect_neurons(self)

    def connect_neurons(self, other_layer):
        """
        :type other_layer: core.layer.Layer
        """
        for neuron in self.neurons:
            for another_neuron in other_layer.neurons:
                if neuron is not another_neuron:
                    neuron.connect_to(another_neuron, 0.1)