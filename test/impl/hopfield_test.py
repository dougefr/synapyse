from core.learning.training_set import TrainingSet
from impl.hopfield import Hopfield
from impl.learning.binary_hebbian_learning import BinaryHebbianLearning
from util import json_util

__author__ = "Douglas Eric Fonseca Rodrigues"

training_set = TrainingSet()

# H
training_set.append([1, 0, 1,
                     1, 1, 1,
                     1, 0, 1])

# T
training_set.append([1, 1, 1,
                     0, 1, 0,
                     0, 1, 0])

hopfield = Hopfield(neuron_count=9, y_high=1, y_low=0)

learning = BinaryHebbianLearning(hopfield)

learning.learn(training_set)

json_util.print_json(hopfield)

# add a incomplete H
training_set.append([1, 0, 0,
                     1, 1, 1,
                     1, 0, 1])

for training_set_row in training_set:
    hopfield.input = training_set_row.input_pattern
    print(hopfield.input)
    hopfield.compute()
    hopfield.compute()
    print(hopfield.output)
