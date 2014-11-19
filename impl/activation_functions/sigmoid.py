from math import exp

from core.activation_functions.activation_function import ActivationFunction


__author__ = 'Douglas Eric Fonseca Rodrigues'


class Sigmoid(ActivationFunction):
    calculated_outputs = {}

    def calculate_output(self, x):
        """
        :type x: float
        """
        if x > 100:
            return 1.0
        elif x < -100:
            return 0.0

        if x in Sigmoid.calculated_outputs:
            return Sigmoid.calculated_outputs[x]
        else:
            output = 1.0 / (1.0 + exp(-1.0 * x))
            Sigmoid.calculated_outputs[x] = output
            return output

    def calculate_derivative(self, x):
        """
        :type x: float
        """
        output = self.calculate_output(x)

        return output * (1.0 - output) + 0.1