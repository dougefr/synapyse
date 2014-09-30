import json

import jsonpickle

from core.learning.error_functions.error_function import ErrorFunction
from core.learning.supervised_learning import SupervisedLearning
from core.learning.training_set import TrainingSet

from core.neural_network import NeuralNetwork
from impl.activation_functions.linear import Linear
from impl.input_functions.weighted_sum import WeightedSum


__author__ = 'Douglas'

neural_network = NeuralNetwork()

input_function = WeightedSum()
activation_function = Linear()

neural_network.add_layer(2, input_function, activation_function)
neural_network.add_layer(3, input_function, activation_function)
neural_network.add_layer(1, input_function, activation_function)

neural_network.randomize_weights()

print(neural_network.compute([1, 2]))

print(json.dumps(json.loads(jsonpickle.encode(neural_network)), indent=4, sort_keys=True))

# learning

training_set = TrainingSet()

training_set.append([0, 0], [0])
training_set.append([0, 1], [1])
training_set.append([1, 0], [1])
training_set.append([1, 1], [0])

supervised_learning = SupervisedLearning(neural_network, ErrorFunction(), 0.01, 10)
supervised_learning.learn(training_set)
