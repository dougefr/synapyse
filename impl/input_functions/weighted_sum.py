from core.input_functions.input_function import InputFunction

__author__ = "Douglas Eric Fonseca Rodrigues"


class WeightedSum(InputFunction):
    def calculate_output(self, input_connections):
        """
        :type input_connections: list[core.connection.Connection]
        """
        s = 0.0

        for input_connection in input_connections:
            s += input_connection.origin.output * input_connection.weight

        return s