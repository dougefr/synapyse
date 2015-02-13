from base.neural_network import NeuralNetwork
from impl.activation_functions.step import Step
from impl.input_functions.weighted_sum import WeightedSum
from impl.multi_layer_perceptron import MultiLayerPerceptron

__author__ = 'Douglas Eric Fonseca Rodrigues'


class Perceptron(MultiLayerPerceptron):
    def __init__(self, input_count, output_count, activation_function=Step(1, 0)):
        NeuralNetwork.__init__(self)
        self.create_layer(input_count, WeightedSum())
        self.create_layer(output_count, WeightedSum(), activation_function)