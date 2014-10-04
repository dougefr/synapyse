from core.learning.training_set import TrainingSet
from impl.activation_functions.tanh import Tanh
from impl.input_functions.weighted_sum import WeightedSum
from impl.learning.back_propagation import BackPropagation
from impl.multi_layer_perceptron import MultiLayerPerceptron
from util import json_util


training_set = TrainingSet()

training_set.append([0.0, 0.0], [0.0])
training_set.append([0.0, 1.0], [1.0])
training_set.append([1.0, 0.0], [1.0])
training_set.append([1.0, 1.0], [0.0])

n = MultiLayerPerceptron()

n.add_layer(2, WeightedSum())
n.add_layer(3, WeightedSum(), Tanh())
n.add_layer(1, WeightedSum(), Tanh())

n.randomize_weights()

b = BackPropagation(n)

b.learn(training_set)

json_util.print_json(n)

print(n.compute([0, 0]))
print(n.compute([0, 1]))
print(n.compute([1, 0]))
print(n.compute([1, 1]))