from core.layer import Layer
from impl.activation_functions.linear import Linear

__author__ = "Douglas Eric Fonseca Rodrigues"


class NeuralNetwork:
    def __init__(self):
        self.layers = []
        """:type : list[core.layer.Layer]"""

    def create_layer(self, neuron_count, input_function, activation_function=Linear()):
        """
        :type neuron_count: int
        :type input_function: core.input_functions.input_function.InputFunction
        :type activation_function: core.activation_functions.activation_function.ActivationFunction
        """
        return self.add_layer(Layer(neuron_count, input_function, activation_function))

    def add_layer(self, new_layer):
        """
        :type new_layer: core.layer.Layer
        """
        if len(self.layers) > 0:
            new_layer.previous = self.layers[-1]

        self.layers.append(new_layer)

        return self

    def remove_layer(self, index):
        del self.layers[index]

        if len(self.layers) > 0:
            for i in range(1, len(self.layers)):
                self.layers[i + 1].previous = self.layers[i]

        return self

    def remove_all_layers(self):
        self.layers.clear()

        return self

    @property
    def output_neurons(self):
        return self.layers[-1].neurons

    def compute(self):
        for layer in self.layers:
            layer.compute_neurons()

        return self

    @property
    def output(self):
        return [neuron.output for neuron in self.output_neurons]

    @property
    def input(self):
        return [neuron.input for neuron in self.layers[0].neurons]

    @input.setter
    def input(self, pattern):
        """
        :type pattern: list[float]
        """
        for neuron, p in zip(self.layers[0].neurons, pattern):
            neuron.input = p

    def set_input(self, pattern):
        """
        :type pattern: list[float]
        """
        self.input = pattern
        return self

    def randomize_weights(self):
        for layer in self.layers:
            layer.randomize_neurons_weights()

        return self

    @property
    def weights(self):
        return [layer.weights for layer in self.layers]