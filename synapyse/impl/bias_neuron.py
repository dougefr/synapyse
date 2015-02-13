from synapyse.base.input_functions.input_function import InputFunction
from synapyse.base.neuron import Neuron

__author__ = 'Douglas Eric Fonseca Rodrigues'


class BiasNeuron(Neuron):
    def __init__(self, activation_function):
        """
        :type activation_function: core.activation_functions.activation_function.ActivationFunction
        """
        Neuron.__init__(self, InputFunction, activation_function)

    def compute_output(self):
        self.output = 1.0
        return self.output