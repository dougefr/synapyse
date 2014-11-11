from core.layer import Layer
from impl.activation_functions.linear import Linear

__author__ = 'Douglas Eric Fonseca Rodrigues'


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
        self.add_layer(Layer(neuron_count, input_function, activation_function))

    def add_layer(self, new_layer):
        """
        :type new_layer: core.layer.Layer
        """
        if len(self.layers) > 0:
            new_layer.previous = self.layers[-1]

        self.layers.append(new_layer)

    def remove_layer(self, index):
        del self.layers[index]

        if len(self.layers) > 0:
            for i in range(1, len(self.layers)):
                self.layers[i + 1].previous = self.layers[i]

    def remove_all_layers(self):
        self.layers.clear()

    @property
    def output_neurons(self):
        return self.layers[-1].neurons

    def compute(self, pattern):
        """
        :type pattern: list[float]
        """

        # sets the input layer with the input_pattern
        for neuron, p in zip(self.layers[0].neurons, pattern):
            neuron.input = p

        for layer in self.layers:
            layer.compute_neurons()

        return [neuron.output for neuron in self.layers[-1].neurons]

    def randomize_weights(self):
        for layer in self.layers:
            layer.randomize_neurons_weights()