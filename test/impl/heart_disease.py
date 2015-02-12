from core.learning.training_set import TrainingSet
from impl.activation_functions.sigmoid import Sigmoid
from impl.input_functions.weighted_sum import WeightedSum
from impl.learning.momentum_back_propagation import MomentumBackPropagation
from impl.multi_layer_perceptron import MultiLayerPerceptron

__author__ = 'Douglas Eric Fonseca Rodrigues'

# Creating a training_set based in a text file
training_set = TrainingSet(13, 1) \
    .import_from_file('heart_disease.txt', ',')\
    .normalize()

# Creating and configuring the network
multi_layer_perceptron = MultiLayerPerceptron() \
    .create_layer(13, WeightedSum()) \
    .create_layer(8, WeightedSum(), Sigmoid()) \
    .create_layer(1, WeightedSum(), Sigmoid()) \
    .randomize_weights()

# Creating and configuring the learning method
momentum_backpropagation = MomentumBackPropagation(neural_network=multi_layer_perceptron,
                                                   learning_rate=0.2,
                                                   momentum=0.6,
                                                   max_error=0.01)

# Configuring a log after each learning method iteration
momentum_backpropagation.on_after_iteration = lambda b: print(b.actual_iteration, ':', b.total_network_error)

# Learning the training_set
momentum_backpropagation.learn(training_set)

# Printing results
for training_set_row in training_set:
    print('Input:', training_set_row.input_pattern)
    print('Ideal output\t: ', training_set_row.ideal_output)

    output = multi_layer_perceptron \
        .set_input(training_set_row.input_pattern) \
        .compute() \
        .output

    print('Resulted output\t: ', output)