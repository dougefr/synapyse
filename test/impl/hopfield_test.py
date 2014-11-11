from core.learning.training_set import TrainingSet
from impl.hopfield import Hopfield
from impl.learning.binary_hebbian_learning import BinaryHebbianLearning
from util import json_util

__author__ = 'Douglas Eric Fonseca Rodrigues'

training_set = TrainingSet()

# H
training_set.append([1, 0, 1,
                     1, 1, 1,
                     1, 0, 1])

# T
training_set.append([1, 1, 1,
                     0, 1, 0,
                     0, 1, 0])

hopfield = Hopfield(9)

learning = BinaryHebbianLearning(hopfield)

learning.learn(training_set)

# add a incomplete H
training_set.append([1, 0, 0,
                     1, 1, 1,
                     1, 0, 1])

json_util.print_json(hopfield)

for training_set_row in training_set:
    print(training_set_row.input_pattern, hopfield.compute(training_set_row.input_pattern))
