from core.layer import Layer

__author__ = 'Douglas'


class NeuralNetwork:
    def __init__(self, learning_method):
        self.layers = []
        """:type : list[Layer]"""

        self.learning_method = learning_method
        """:type : LearningMethod"""

    def add_layer(self, neuron_count, input_function, activation_function):
        """
        :type neuron_count: int
        :type input_function: InputFunction
        :type activation_function:
        """
        self._add_layer(Layer(neuron_count, input_function, activation_function))

    def _add_layer(self, new_layer):
        """
        :type new_layer: Layer
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

        # sets the input layer with the pattern
        for neuron, p in zip(self.layers[0].neurons, pattern):
            neuron.input = p

        for layer in self.layers:
            layer.compute_neurons()

        return [neuron.output for neuron in self.layers[-1].neurons]

    def randomize_weights(self):
        for layer in self.layers:
            layer.randomize_neurons_weights()

    def learn(self, training_set):
        """
        :type training_set: TrainingSet
        """
        self.learning_method.learn(training_set)