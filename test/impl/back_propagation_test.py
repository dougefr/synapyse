from core.learning.training_set import TrainingSet
from impl.activation_functions.tanh import Tanh
from impl.input_functions.weighted_sum import WeightedSum
from impl.learning.back_propagation import BackPropagation
from impl.multi_layer_perceptron import MultiLayerPerceptron


__author__ = 'Douglas Eric Fonseca Rodrigues'

training_set = TrainingSet(2, 1)

training_set \
    .append([0.0, 0.0], [0.0]) \
    .append([0.0, 1.0], [1.0]) \
    .append([1.0, 0.0], [1.0]) \
    .append([1.0, 1.0], [0.0])

n = MultiLayerPerceptron()

n \
    .create_layer(2, WeightedSum()) \
    .create_layer(3, WeightedSum(), Tanh(2)) \
    .create_layer(1, WeightedSum(), Tanh(2))

n.randomize_weights()

b = BackPropagation(n, learning_rate=0.1, max_error=0.01)

b.on_after_iteration = lambda x: print(x.actual_iteration, ':', x.total_network_error)

b.learn(training_set)

for training_set_row in training_set:
    print(n.set_input(training_set_row.input_pattern)
          .compute()
          .output)