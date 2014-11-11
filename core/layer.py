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
        self.neurons = [Neuron(input_function, activation_function) for _ in range(neuron_count)]

        self.__previous = None
        """:type : core.layer.Layer"""

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

    def compute_neurons(self):
        return [neuron.compute_output() for neuron in self.neurons]

    def randomize_neurons_weights(self):
        for neuron in self.neurons:
            neuron.randomize_weights()