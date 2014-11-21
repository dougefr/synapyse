from core.learning.training_set import TrainingSet
from impl.activation_functions.sigmoid import Sigmoid
from impl.input_functions.weighted_sum import WeightedSum
from impl.learning.momentum_back_propagation import MomentumBackPropagation
from impl.multi_layer_perceptron import MultiLayerPerceptron

__author__ = "Douglas Eric Fonseca Rodrigues"

training_set = TrainingSet()

training_set \
    .append([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0]) \
    .append([1.0, 0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0]) \
    .append([1.0, 1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 1.0]) \
    .append([1.0, 1.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0]) \
    .append([1.0, 0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 1.0]) \
    .append([1.0, 1.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0, 0.0]) \
    .append([1.0, 1.0, 1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0, 1.0]) \
    .append([1.0, 0.0, 1.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0]) \
    .append([0.0, 1.0, 1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 1.0]) \
    .append([0.0, 1.0, 1.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 1.0, 0.0]) \
    .append([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0, 1.0]) \
    .append([1.0, 0.0, 1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 1.0, 0.0, 0.0]) \
    .append([1.0, 1.0, 0.0, 0.0, 1.0, 0.0], [0.0, 1.0, 1.0, 0.0, 1.0]) \
    .append([1.0, 1.0, 0.0, 1.0, 1.0, 0.0], [0.0, 1.0, 1.0, 1.0, 0.0]) \
    .append([1.0, 0.0, 0.0, 1.0, 1.0, 0.0], [0.0, 1.0, 1.0, 1.0, 1.0]) \
    .append([1.0, 1.0, 1.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0]) \
    .append([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 0.0, 0.0, 0.0, 1.0]) \
    .append([1.0, 0.0, 1.0, 1.0, 1.0, 0.0], [1.0, 0.0, 0.0, 1.0, 0.0]) \
    .append([0.0, 1.0, 1.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 1.0, 1.0]) \
    .append([0.0, 1.0, 1.0, 1.0, 1.0, 0.0], [1.0, 0.0, 1.0, 0.0, 0.0]) \
    .append([1.0, 0.0, 0.0, 0.0, 1.0, 1.0], [1.0, 0.0, 1.0, 0.0, 1.0]) \
    .append([1.0, 0.0, 1.0, 0.0, 1.0, 1.0], [1.0, 0.0, 1.0, 1.0, 0.0]) \
    .append([0.0, 1.0, 1.0, 1.0, 1.0, 0.0], [1.0, 0.0, 1.0, 1.0, 1.0]) \
    .append([1.0, 1.0, 0.0, 0.0, 1.0, 1.0], [1.0, 1.0, 0.0, 0.0, 0.0]) \
    .append([1.0, 1.0, 0.0, 1.0, 1.0, 1.0], [1.0, 1.0, 0.0, 0.0, 1.0]) \
    .append([1.0, 0.0, 0.0, 1.0, 1.0, 1.0], [1.0, 1.0, 0.0, 1.0, 0.0])

neural_network = MultiLayerPerceptron()

neural_network \
    .create_layer(6, WeightedSum()) \
    .create_layer(7, WeightedSum(), Sigmoid()) \
    .create_layer(5, WeightedSum(), Sigmoid()) \
    .randomize_weights()

b = MomentumBackPropagation(neural_network=neural_network,
                            learning_rate=0.1,
                            momentum=0.4,
                            max_error=0.02)

b.on_after_iteration = lambda obj: print(obj.actual_iteration, ":", obj.total_network_error)

b.learn(training_set)

print("total_error=", b.total_network_error)

for row in training_set:
    actual = neural_network \
        .set_input(row.input_pattern) \
        .compute() \
        .output

    print(row.input_pattern, "actual=", actual, "ideal=", row.ideal_output)
