from core.activation_functions.activation_function import ActivationFunction

__author__ = 'Douglas Eric Fonseca Rodrigues'


class Step(ActivationFunction):
    def __init__(self, y_high=1, y_low=0):
        """
        :type y_high: float
        :type y_low: float
        """
        self.y_high = y_high
        self.y_low = y_low

    def calculate_output(self, x):
        """
        :type x: float
        """
        if x > 0:
            return self.y_high
        else:
            return self.y_low

    def calculate_derivative(self, x):
        """
        :type x: float
        """
        return 1