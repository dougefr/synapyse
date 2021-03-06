from synapyse.base.learning.training_set import TrainingSet
from synapyse.impl.learning.least_mean_square import LeastMeanSquare
from synapyse.impl.perceptron import Perceptron


__author__ = 'Douglas'

# Logger.enable_logger(Logger.DEBUG)

training_set = TrainingSet(2, 2)

training_set \
    .append([0.0, 0.0], [0.0, 0.0]) \
    .append([0.0, 1.0], [0.0, 1.0]) \
    .append([1.0, 0.0], [0.0, 1.0]) \
    .append([1.0, 1.0], [1.0, 1.1])

neural_network = Perceptron(2, 2) \
    .randomize_weights()

lms = LeastMeanSquare(neural_network, 0.5, 0.1)
lms.on_after_iteration = lambda obj: print(obj.actual_iteration, ':', obj.total_network_error)
lms.learn(training_set)

for row in training_set:
    actual = neural_network \
        .set_input(row.input_pattern) \
        .compute() \
        .output

    print(row.input_pattern, 'actual=', actual, 'ideal=', row.ideal_output)
