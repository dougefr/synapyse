from core.learning.training_set import TrainingSet
from impl.activation_functions.sigmoid import Sigmoid
from impl.input_functions.weighted_sum import WeightedSum
from impl.learning.momentum_back_propagation import MomentumBackPropagation
from impl.multi_layer_perceptron import MultiLayerPerceptron

__author__ = "Douglas Eric Fonseca Rodrigues"

# Creating a training_set based in a text file
training_set = TrainingSet() \
    .import_from_file("car_evaluation.txt", 21, 4, ",")

# Creating and configuring the network
multi_layer_perceptron = MultiLayerPerceptron() \
    .create_layer(21, WeightedSum()) \
    .create_layer(14, WeightedSum(), Sigmoid()) \
    .create_layer(4, WeightedSum(), Sigmoid()) \
    .randomize_weights()

# Creating and configuring the learning method
momentum_backpropagation = MomentumBackPropagation(neural_network=multi_layer_perceptron,
                                                   learning_rate=0.3,
                                                   momentum=0.6,
                                                   max_error=0.01)

# Configuring a log after each learning method iteration
momentum_backpropagation.on_after_iteration = lambda b: print(b.actual_iteration, ":", b.total_network_error)

# Learning the training_set
momentum_backpropagation.learn(training_set)

# Printing results
for training_set_row in training_set:
    print("Input:", training_set_row.input_pattern)
    print("Ideal output\t: ", training_set_row.ideal_output)

    output = multi_layer_perceptron \
        .set_input(training_set_row.input_pattern).compute() \
        .output

    print("Resulted output\t: ", output)
