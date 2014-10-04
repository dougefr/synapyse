from math import exp

from core.activation_functions.activation_function import ActivationFunction


__author__ = 'Douglas Eric Fonseca Rodrigues'


class Sigmoid(ActivationFunction):
    def calculate_output(self, x):
        """
        :type x: float
        """
        if x > 100:
            return 1.0
        elif x < -100:
            return 0.0

        return 1.0 / (1 + exp(-1.0 * x))

    def calculate_derivative(self, x):
        """
        :type x: float
        """
        return x * (1.0 - x) + 0.1