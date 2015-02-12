from core.neuron import Neuron


__author__ = 'Douglas Eric Fonseca Rodrigues'


class Layer:
    def __init__(self, neuron_count, input_function, activation_function):
        """
        :type neuron_count: int
        :type input_function: core.input_functions.input_function.InputFunction
        :type activation_function: core.activation_functions.activation_function.ActivationFunction
        """
        self.input_function = input_function
        self.activation_function = activation_function
        self.neurons = [self.instantiate_neurons() for _ in range(neuron_count)]
        """:type : list[core.neuron.Neuron]"""

        self.__previous = None
        """:type : core.layer.Layer"""

    def instantiate_neurons(self):
        return Neuron(self.input_function, self.activation_function)

    @property
    def previous(self):
        return self.__previous

    @previous.setter
    def previous(self, previous_layer):
        """
        :type previous_layer: core.layer.Layer
        """
        self.__previous = previous_layer
        self.connect_neurons(previous_layer)

    def connect_neurons(self, other_layer):
        """
        :type other_layer: core.layer.Layer
        """
        for neuron in self.neurons:
            for another_neuron in other_layer.neurons:
                neuron.connect_to(another_neuron)

        return self

    def compute_neurons(self):
        for neuron in self.neurons:
            neuron.compute_output()

        return self

    def randomize_neurons_weights(self):
        for neuron in self.neurons:
            neuron.randomize_weights()

        return self

    @property
    def input(self):
        return [neuron.input for neuron in self.neurons]

    @input.setter
    def input(self, pattern):
        """
        :type pattern: list[float]
        """
        for neuron, p in zip(self.neurons, pattern):
            neuron.input = p

    @property
    def weights(self):
        return [neuron.weights for neuron in self.neurons]
