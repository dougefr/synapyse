from core.learning.training_set import TrainingSet
from impl.activation_functions.tanh import Tanh
from impl.input_functions.weighted_sum import WeightedSum
from impl.learning.back_propagation import BackPropagation
from impl.multi_layer_perceptron import MultiLayerPerceptron
from util import json_util


__author__ = 'Douglas Eric Fonseca Rodrigues'

training_set = TrainingSet()

training_set.append([0.0, 0.0], [0.0])
training_set.append([0.0, 1.0], [1.0])
training_set.append([1.0, 0.0], [1.0])
training_set.append([1.0, 1.0], [0.0])

n = MultiLayerPerceptron()

n.create_layer(2, WeightedSum())
n.create_layer(3, WeightedSum(), Tanh())
n.create_layer(1, WeightedSum(), Tanh())

n.randomize_weights()

b = BackPropagation(n, max_error=0.001)

b.on_total_error_calculate = lambda x: print(x)

b.learn(training_set)

json_util.print_json(n)

print(n.weights)

for training_set_row in training_set:
    n.input = training_set_row.input_pattern
    n.compute()
    print(n.output)