import json

import jsonpickle

from core.activation_functions.activation_function import ActivationFunction
from core.input_functions.input_function import InputFunction
from core.learning.supervised_learning import SupervisedLearning
from core.learning.training_set import TrainingSet
from core.neural_network import NeuralNetwork


__author__ = 'Douglas Eric Fonseca Rodrigues'

neural_network = NeuralNetwork() \
    .create_layer(2, InputFunction(), ActivationFunction()) \
    .create_layer(3, InputFunction(), ActivationFunction()) \
    .create_layer(1, InputFunction(), ActivationFunction()) \
    .randomize_weights()

neural_network.input = [1, 2]

print(neural_network.compute())

print(json.dumps(json.loads(jsonpickle.encode(neural_network)), indent=4, sort_keys=True))

# learning

training_set = TrainingSet()

training_set.append([0, 0], [0]) \
    .append([0, 1], [1]) \
    .append([1, 0], [1]) \
    .append([1, 1], [0])

supervised_learning = SupervisedLearning(neural_network=neural_network,
                                         learning_rate=0.1,
                                         max_error=0.1,
                                         max_iterations=1)
supervised_learning.learn(training_set)
