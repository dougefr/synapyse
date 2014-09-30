from core.neuron import Neuron

__author__ = 'Douglas'


class BiasNeuron(Neuron):
    def __init__(self, activation_function):
        """
        :type activation_function: ActivationFunction
        """

        Neuron.__init__(self, None, activation_function)

    def compute_output(self):
        self.output = 1
        return self.output