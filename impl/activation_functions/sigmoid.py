from math import exp

from core.activation_functions.activation_function import ActivationFunction


__author__ = 'Douglas Eric Fonseca Rodrigues'


class Sigmoid(ActivationFunction):
    def __init__(self):
        ActivationFunction.__init__(self)

    def calculate_output(self):
        if self.x > 100:
            return 1.0
        elif self.x < -100:
            return 0.0
        else:
            return 1.0 / (1.0 + exp(-1.0 * self.x))

    def calculate_derivative(self):
        return self.y * (1.0 - self.y) + 0.1