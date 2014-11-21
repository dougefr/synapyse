from core.connection import Connection


__author__ = "Douglas Eric Fonseca Rodrigues"


class Neuron:
    def __init__(self, input_function, activation_function):
        """
        :type input_function: core.input_functions.input_function.InputFunction
        :type activation_function: core.activation_functions.activation_function.ActivationFunction
        """
        self.input_function = input_function
        self.activation_function = activation_function.clone()
        self.output = 0.0
        self._input = 0.0

        self.input_connections = InputConnectionDict()

    def compute_output(self):
        if not self.input_connections.is_empty():
            self.input = self.input_function.calculate_output(self.input_connections.values())

        self.output = self.activation_function.set_x(self.input).y

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


class InputConnectionDict:
    def __init__(self):
        self.dict = {}
        """:type : dict[core.neuron.Neuron, core.connection.Connection]"""

        self.list = []
        """:type : list[core.connection.Connection]"""

        self.__empty = True

    def __setitem__(self, key, value):
        """
        :type key: core.neuron.Neuron
        :type value: core.connection.Connection
        """
        if key in self.dict:
            self.__remove_item_list(key)

        self.dict[key] = value
        self.list.append(value)
        self.__empty = False

    def __getitem__(self, key):
        """
        :type key: core.neuron.Neuron
        """
        return self.dict[key]

    def __delitem__(self, key):
        """
        :type key: core.neuron.Neuron
        """
        del self.dict[key]
        self.__remove_item_list(key)

    def values(self):
        return self.list

    def __remove_item_list(self, key):
        """
        :type key: core.neuron.Neuron
        """
        for item in self.list:
            if item == self.dict[key]:
                self.list.remove(item)

                if not self.list:
                    self.__empty = True

                break

    def is_empty(self):
        return self.__empty

    def clear(self):
        self.dict.clear()
        self.list.clear()
        self.__empty = True