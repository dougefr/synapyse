from math import tanh, cosh

from core.activation_functions.activation_function import ActivationFunction


__author__ = 'douglas'


class Tanh(ActivationFunction):
    def calculate_output(self, x):
        """
        :type x: float
        """
        if x > 100:
            return 1
        elif x < -100:
            return -1
        else:
            return tanh(x)

    def calculate_derivative(self, x):
        """
        :type x: float
        """
        if x < -6:
            return self.calculate_derivative(-6)
        elif x > 6:
            return self.calculate_derivative(6)
        else:
            return (1.0 / cosh(x)) ** 2