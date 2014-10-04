from core.learning.error_functions.error_function import ErrorFunction

__author__ = 'Douglas'


class RMS(ErrorFunction):
    def __init__(self):
        self.global_error = 0.0
        """:type : float"""

        self.size = 0
        """:type : int"""

    @property
    def total_error(self):
        return self.global_error / self.size

    def add_error(self, output_error):
        """
        :type output_error: list[float]
        """
        self.global_error += sum([(error * error) * 0.5 for error in output_error])
        self.size += len(output_error)

    def reset(self):
        self.global_error = 0.0
        self.size = 0.0