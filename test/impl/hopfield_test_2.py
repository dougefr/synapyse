from core.learning.training_set import TrainingSet
from impl.hopfield import Hopfield
from impl.learning.binary_hebbian_learning import BinaryHebbianLearning

__author__ = 'Douglas Eric Fonseca Rodrigues'


def print_matrix(v):
    for i in range(len(v)):
        if i % 10 == 0:
            print('\n', end='')

        if v[i] == 1:
            print('O', end='')
        else:
            print(' ', end='')

    print('\n')


training_set = TrainingSet(100) \
    .append([
    1, -1, 1, -1, 1, -1, 1, -1, 1, -1,
    -1, 1, -1, 1, -1, 1, -1, 1, -1, 1,
    1, -1, 1, -1, 1, -1, 1, -1, 1, -1,
    -1, 1, -1, 1, -1, 1, -1, 1, -1, 1,
    1, -1, 1, -1, 1, -1, 1, -1, 1, -1,
    -1, 1, -1, 1, -1, 1, -1, 1, -1, 1,
    1, -1, 1, -1, 1, -1, 1, -1, 1, -1,
    -1, 1, -1, 1, -1, 1, -1, 1, -1, 1,
    1, -1, 1, -1, 1, -1, 1, -1, 1, -1,
    -1, 1, -1, 1, -1, 1, -1, 1, -1, 1
]) \
    .append([
    1, 1, -1, -1, 1, 1, -1, -1, 1, 1,
    1, 1, -1, -1, 1, 1, -1, -1, 1, 1,
    -1, -1, 1, 1, -1, -1, 1, 1, -1, -1,
    -1, -1, 1, 1, -1, -1, 1, 1, -1, -1,
    1, 1, -1, -1, 1, 1, -1, -1, 1, 1,
    1, 1, -1, -1, 1, 1, -1, -1, 1, 1,
    -1, -1, 1, 1, -1, -1, 1, 1, -1, -1,
    -1, -1, 1, 1, -1, -1, 1, 1, -1, -1,
    1, 1, -1, -1, 1, 1, -1, -1, 1, 1,
    1, 1, -1, -1, 1, 1, -1, -1, 1, 1
]) \
    .append([
    1, 1, 1, 1, 1, -1, -1, -1, -1, -1,
    1, 1, 1, 1, 1, -1, -1, -1, -1, -1,
    1, 1, 1, 1, 1, -1, -1, -1, -1, -1,
    1, 1, 1, 1, 1, -1, -1, -1, -1, -1,
    1, 1, 1, 1, 1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, 1, 1, 1, 1, 1,
    -1, -1, -1, -1, -1, 1, 1, 1, 1, 1,
    -1, -1, -1, -1, -1, 1, 1, 1, 1, 1,
    -1, -1, -1, -1, -1, 1, 1, 1, 1, 1,
    -1, -1, -1, -1, -1, 1, 1, 1, 1, 1,
]) \
    .append([
    1, -1, -1, 1, -1, -1, 1, -1, -1, 1,
    -1, 1, -1, -1, 1, -1, -1, 1, -1, -1,
    -1, -1, 1, -1, -1, 1, -1, -1, 1, -1,
    1, -1, -1, 1, -1, -1, 1, -1, -1, 1,
    -1, 1, -1, -1, 1, -1, -1, 1, -1, -1,
    -1, -1, 1, -1, -1, 1, -1, -1, 1, -1,
    1, -1, -1, 1, -1, -1, 1, -1, -1, 1,
    -1, 1, -1, -1, 1, -1, -1, 1, -1, -1,
    -1, -1, 1, -1, -1, 1, -1, -1, 1, -1,
    1, -1, -1, 1, -1, -1, 1, -1, -1, 1
]) \
    .append([
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, -1, -1, -1, -1, -1, -1, -1, -1, 1,
    1, -1, 1, 1, 1, 1, 1, 1, -1, 1,
    1, -1, 1, -1, -1, -1, -1, 1, -1, 1,
    1, -1, 1, -1, 1, 1, -1, 1, -1, 1,
    1, -1, 1, -1, 1, 1, -1, 1, -1, 1,
    1, -1, 1, -1, -1, -1, -1, 1, -1, 1,
    1, -1, 1, 1, 1, 1, 1, 1, -1, 1,
    1, -1, -1, -1, -1, -1, -1, -1, -1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1
])

hopfield = Hopfield(100, 1, -1)

learning = BinaryHebbianLearning(hopfield)

learning.learn(training_set)

# Add some 'damaged' data
training_set.append([
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, 1, -1, 1, -1, 1, -1, 1, -1, 1,
    1, -1, 1, -1, 1, -1, 1, -1, 1, -1,
    -1, 1, -1, 1, -1, 1, -1, 1, -1, 1,
    1, -1, 1, -1, 1, -1, 1, -1, 1, -1,
    -1, 1, -1, 1, -1, 1, -1, 1, -1, 1
]) \
    .append([
    1, 1, 1, -1, 1, -1, -1, -1, -1, 1,
    -1, 1, -1, -1, 1, 1, 1, -1, 1, 1,
    -1, -1, 1, -1, 1, -1, 1, 1, -1, 1,
    -1, 1, 1, 1, -1, -1, -1, 1, -1, -1,
    1, 1, -1, -1, 1, -1, -1, 1, 1, 1,
    -1, 1, -1, 1, 1, 1, -1, -1, -1, 1,
    1, -1, 1, 1, -1, -1, 1, -1, -1, 1,
    -1, -1, -1, 1, -1, 1, 1, 1, -1, -1,
    1, 1, -1, 1, 1, 1, -1, -1, 1, -1,
    -1, 1, -1, -1, 1, -1, -1, 1, 1, 1
]) \
    .append([
    1, 1, 1, 1, 1, -1, -1, -1, -1, -1,
    1, -1, -1, -1, 1, -1, 1, 1, 1, -1,
    1, -1, -1, -1, 1, -1, 1, 1, 1, -1,
    1, -1, -1, -1, 1, -1, 1, 1, 1, -1,
    1, 1, 1, 1, 1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, 1, 1, 1, 1, 1,
    -1, 1, 1, 1, -1, 1, -1, -1, -1, 1,
    -1, 1, 1, 1, -1, 1, -1, -1, -1, 1,
    -1, 1, 1, 1, -1, 1, -1, -1, -1, 1,
    -1, -1, -1, -1, -1, 1, 1, 1, 1, 1
]) \
    .append([
    1, -1, -1, 1, 1, 1, 1, -1, -1, 1,
    1, 1, -1, -1, 1, 1, 1, 1, -1, -1,
    1, 1, 1, -1, -1, 1, 1, 1, 1, -1,
    1, 1, 1, 1, -1, -1, 1, 1, 1, 1,
    -1, 1, 1, 1, 1, -1, -1, 1, 1, 1,
    -1, -1, 1, 1, 1, 1, -1, -1, 1, 1,
    1, -1, -1, 1, 1, 1, 1, -1, -1, 1,
    1, 1, -1, -1, 1, 1, 1, 1, -1, -1,
    1, 1, 1, -1, -1, 1, 1, 1, 1, -1,
    1, 1, 1, 1, -1, -1, 1, 1, 1, 1
]) \
    .append([
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, -1, -1, -1, -1, -1, -1, -1, -1, 1,
    1, -1, -1, -1, -1, -1, -1, -1, -1, 1,
    1, -1, -1, -1, -1, -1, -1, -1, -1, 1,
    1, -1, -1, -1, 1, 1, -1, -1, -1, 1,
    1, -1, -1, -1, 1, 1, -1, -1, -1, 1,
    1, -1, -1, -1, -1, -1, -1, -1, -1, 1,
    1, -1, -1, -1, -1, -1, -1, -1, -1, 1,
    1, -1, -1, -1, -1, -1, -1, -1, -1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1
])

# json_util.print_json(hopfield)

for training_set_row in training_set:
    hopfield.input = training_set_row.input_pattern
    print('Pattern:')
    print_matrix(hopfield.input)
    hopfield.compute()
    hopfield.compute()
    print('Result:')
    print_matrix(hopfield.output)
