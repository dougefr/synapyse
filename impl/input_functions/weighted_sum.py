from core.input_functions.input_function import InputFunction

__author__ = 'Douglas Eric Fonseca Rodrigues'


class WeightedSum(InputFunction):
    def calculate_output(self, input_connections):
        """
        :type input_connections: list[core.connection.Connection]
        """
        return sum([input_connection.input * input_connection.weight for input_connection in input_connections])