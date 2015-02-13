from base.neuron import Neuron

__author__ = 'Douglas Eric Fonseca Rodrigues'


class InputOutputNeuron(Neuron):
    def __init__(self, input_function, activation_function, bias=0.0):
        """
        :type input_function: core.input_functions.input_function.InputFunction
        :type activation_function: core.activation_functions.activation_function.ActivationFunction
        :type bias: float
        """
        Neuron.__init__(self, input_function, activation_function)

        self.bias = bias
        self.external_input = False

    @property
    def input(self):
        return self._input

    @input.setter
    def input(self, n):
        self.external_input = True
        self._input = n

    def compute_output(self):
        if not self.external_input:
            if not self.input_connections.is_empty():
                self.input = self.input_function.calculate_output(list(self.input_connections.values()))

        self.output = self.activation_function.set_x(self.input + self.bias).y

        if self.external_input:
            self.external_input = False
            self._input = 0.0