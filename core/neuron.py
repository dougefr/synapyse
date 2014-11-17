from core.connection import Connection


__author__ = 'Douglas Eric Fonseca Rodrigues'


class Neuron:
    def __init__(self, input_function, activation_function):
        """
        :type input_function: core.input_functions.input_function.InputFunction
        :type activation_function: core.activation_functions.activation_function.ActivationFunction
        """
        self.input_function = input_function
        self.activation_function = activation_function
        self.output = 0.0
        self._input = 0.0

        self.input_connections = {}
        """:type : dict[core.neuron.Neuron, core.connection.Connection]"""

    def compute_output(self):
        if len(self.input_connections) > 0:
            self.input = self.input_function.calculate_output(list(self.input_connections.values()))

        self.output = self.activation_function.calculate_output(self.input)

        return self.output

    def randomize_weights(self):
        for input_connection in self.input_connections.values():
            input_connection.randomize_weight()

    def connect_to(self, another_neuron, weight=0.0):
        """
        :type another_neuron: core.neuron.Neuron
        :type weight: float
        """
        connection = Connection(another_neuron, self, weight)

        self.input_connections[another_neuron] = connection

    def disconnect_to(self, another_neuron):
        """
        :type another_neuron: core.neuron.Neuron
        """
        del self.input_connections[another_neuron]

    def remove_all_connections(self):
        self.input_connections.clear()

    @property
    def input(self):
        return self._input

    @input.setter
    def input(self, n):
        """
        :type n: float
        """
        self._input = n

    @property
    def weights(self):
        return [input_connection.weight for input_connection in self.input_connections.values()]