import logging
import sys
from base.learning.training_set import TrainingSet
from impl.learning.least_mean_square import LeastMeanSquare
from impl.perceptron import Perceptron

__author__ = 'Douglas'

logger = logging.getLogger('synapyse')
#logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)
logger.addHandler(ch)



training_set = TrainingSet(2, 1)

training_set \
    .append([0.0, 0.0], [0.0]) \
    .append([0.0, 1.0], [0.0]) \
    .append([1.0, 0.0], [0.0]) \
    .append([1.0, 1.0], [1.0])

neural_network = Perceptron(2, 1)\
    .randomize_weights()

lms = LeastMeanSquare(neural_network, 0.5, 0)
lms.on_after_iteration = lambda obj: print(obj.actual_iteration, ':', obj.total_network_error)
lms.learn(training_set)

for row in training_set:
    actual = neural_network \
        .set_input(row.input_pattern) \
        .compute() \
        .output

    print(row.input_pattern, 'actual=', actual, 'ideal=', row.ideal_output)
