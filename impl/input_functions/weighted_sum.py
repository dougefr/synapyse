from core.input_functions.input_function import InputFunction

__author__ = 'Douglas'


class WeightedSum(InputFunction):
    def calculate_output(self, input_connections):
        """
        :type input_connections: list[Connection]
        """
        return sum([input_connection.origin.output * input_connection.weight for input_connection in input_connections])