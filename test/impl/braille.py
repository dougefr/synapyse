from core.learning.training_set import TrainingSet
from impl.activation_functions.tanh import Tanh
from impl.input_functions.weighted_sum import WeightedSum
from impl.learning.momentum_backpropagation import MomentumBackpropagation
from impl.multi_layer_perceptron import MultiLayerPerceptron

__author__ = 'Douglas Eric Fonseca Rodrigues'

training_set = TrainingSet()

training_set.append([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0])  # A
training_set.append([1.0, 0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0])  # B
training_set.append([1.0, 1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 1.0])  # C
training_set.append([1.0, 1.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0])  # D
training_set.append([1.0, 0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 1.0])  # E
training_set.append([1.0, 1.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0, 0.0])  # F
training_set.append([1.0, 1.0, 1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0, 1.0])  # G
training_set.append([1.0, 0.0, 1.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0])  # H
training_set.append([0.0, 1.0, 1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 1.0])  # I
training_set.append([0.0, 1.0, 1.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 1.0, 0.0])  # J
training_set.append([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0, 1.0])  # K
training_set.append([1.0, 0.0, 1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 1.0, 0.0, 0.0])  # L
training_set.append([1.0, 1.0, 0.0, 0.0, 1.0, 0.0], [0.0, 1.0, 1.0, 0.0, 1.0])  # M
training_set.append([1.0, 1.0, 0.0, 1.0, 1.0, 0.0], [0.0, 1.0, 1.0, 1.0, 0.0])  # N
training_set.append([1.0, 0.0, 0.0, 1.0, 1.0, 0.0], [0.0, 1.0, 1.0, 1.0, 1.0])  # O
training_set.append([1.0, 1.0, 1.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0])  # P
training_set.append([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 0.0, 0.0, 0.0, 1.0])  # Q
training_set.append([1.0, 0.0, 1.0, 1.0, 1.0, 0.0], [1.0, 0.0, 0.0, 1.0, 0.0])  # R
training_set.append([0.0, 1.0, 1.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 1.0, 1.0])  # S
training_set.append([0.0, 1.0, 1.0, 1.0, 1.0, 0.0], [1.0, 0.0, 1.0, 0.0, 0.0])  # T
training_set.append([1.0, 0.0, 0.0, 0.0, 1.0, 1.0], [1.0, 0.0, 1.0, 0.0, 1.0])  # U
training_set.append([1.0, 0.0, 1.0, 0.0, 1.0, 1.0], [1.0, 0.0, 1.0, 1.0, 0.0])  # V
training_set.append([0.0, 1.0, 1.0, 1.0, 1.0, 0.0], [1.0, 0.0, 1.0, 1.0, 1.0])  # W
training_set.append([1.0, 1.0, 0.0, 0.0, 1.0, 1.0], [1.0, 1.0, 0.0, 0.0, 0.0])  # X
training_set.append([1.0, 1.0, 0.0, 1.0, 1.0, 1.0], [1.0, 1.0, 0.0, 0.0, 1.0])  # Y
training_set.append([1.0, 0.0, 0.0, 1.0, 1.0, 1.0], [1.0, 1.0, 0.0, 1.0, 0.0])  # Z

n = MultiLayerPerceptron()

n.create_layer(6, WeightedSum())
n.create_layer(6, WeightedSum(), Tanh())
n.create_layer(5, WeightedSum(), Tanh())

n.randomize_weights()

b = MomentumBackpropagation(n,
                            learning_rate=0.4,
                            momentum=0.5,
                            max_error=0.01,
                            max_iterations=5000)

b.on_after_iteration = lambda obj: print(obj.total_network_error)

b.learn(training_set)

print("total_error=", b.total_network_error)

for row in training_set:
    n.input = row.input_pattern
    n.compute()
    actual = n.output
    print(row.input_pattern, "actual=", actual, "ideal=", row.ideal_output)
