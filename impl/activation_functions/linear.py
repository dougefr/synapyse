from core.activation_functions.activation_function import ActivationFunction

__author__ = 'Douglas Eric Fonseca Rodrigues'


class Linear(ActivationFunction):
    def calculate_output(self):
        return self.x

    def calculate_derivative(self):
        return 1.0