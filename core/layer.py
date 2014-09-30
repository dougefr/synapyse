from core.neuron import Neuron

__author__ = 'Douglas'


class Layer:
    def __init__(self, neuron_count, input_function, activation_function):
        """
        :type neuron_count: int
        :type input_function: InputFunction
        :type activation_function: ActivationFunction
        """

        self.neurons = [Neuron(input_function, activation_function) for _ in range(neuron_count)]
        """:type : list[Neuron]"""

        self.__previous = None

    @property
    def previous(self):
        return self.__previous

    @previous.setter
    def previous(self, previous_layer):
        """
        :type previous_layer: Layer
        """
        self.__previous = previous_layer

        for neuron in self.neurons:
            for another_neuron in previous_layer.neurons:
                neuron.connect_to(another_neuron)

    def compute_neurons(self):
        """
        :type pattern: list[float]
        """
        return [neuron.compute_output() for neuron in self.neurons]

    def randomize_neurons_weights(self):
        for neuron in self.neurons:
            neuron.randomize_weights()