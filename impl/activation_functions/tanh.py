from math import exp

from core.activation_functions.activation_function import ActivationFunction


__author__ = 'Douglas Eric Fonseca Rodrigues'


class Tanh(ActivationFunction):
    def __init__(self, slope):
        """
        :type slope: float
        """
        self.slope = slope
        ActivationFunction.__init__(self)

    def calculate_output(self):
        if self.x > 100:
            return 1
        elif self.x < -100:
            return -1
        else:
            e_x = exp(self.slope * self.x)
            return (e_x - 1.0) / (e_x + 1.0)

    def calculate_derivative(self):
        x = self.y * self.y
        return 1.0 - x

    def clone(self):
        clone = Tanh(self.slope)
        clone.x = self.x
        return clone