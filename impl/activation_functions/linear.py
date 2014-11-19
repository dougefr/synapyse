from core.activation_functions.activation_function import ActivationFunction

__author__ = 'Douglas Eric Fonseca Rodrigues'


class Linear(ActivationFunction):
    def calculate_output(self, x):
        """
        :type x: float
        """
        return x

    def calculate_derivative(self, x):
        """
        :type x: float
        """
        return 1.0