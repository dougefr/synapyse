from core.layer import Layer
from impl.activation_functions.step import Step
from impl.input_functions.weighted_sum import WeightedSum
from impl.input_output_neuron import InputOutputNeuron

__author__ = 'Douglas Eric Fonseca Rodrigues'


class HopfieldLayer(Layer):
    def __init__(self, neuron_count, y_high, y_low):
        """
        :type neuron_count: int
        :type y_high: float
        :type y_low: float
        """
        Layer.__init__(self, neuron_count, WeightedSum(), Step(y_high, y_low))
        self.connect_neurons(self)

    def instantiate_neurons(self):
        return InputOutputNeuron(self.input_function, self.activation_function)

    def connect_neurons(self, other_layer):
        """
        :type other_layer: core.layer.Layer
        """
        for neuron in self.neurons:
            for another_neuron in other_layer.neurons:
                if neuron is not another_neuron:
                    neuron.connect_to(another_neuron)