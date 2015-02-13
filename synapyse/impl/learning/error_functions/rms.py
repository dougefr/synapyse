from synapyse.base.learning.error_functions.error_function import ErrorFunction

__author__ = 'Douglas Eric Fonseca Rodrigues'


class RMS(ErrorFunction):
    def __init__(self, n):
        self.global_error = 0.0
        self.n = n

    @property
    def total_error(self):
        return self.global_error / self.n

    def add_error(self, output_error):
        """
        :type output_error: list[float]
        """
        for error in output_error:
            self.global_error += (error * error) * 0.5

    def reset(self):
        self.global_error = 0.0