from synapyse.base.learning.training_set import TrainingSet
from synapyse.impl.activation_functions.sigmoid import Sigmoid
from synapyse.impl.input_functions.weighted_sum import WeightedSum
from synapyse.impl.learning.momentum_back_propagation import MomentumBackPropagation
from synapyse.impl.multi_layer_perceptron import MultiLayerPerceptron
from synapyse.util.logger import Logger

__author__ = 'Douglas Eric Fonseca Rodrigues'

Logger.enable_logger(Logger.INFO)
sim = 0.3

# Creating a training_set based in a text file
training_set_training, training_set_test = TrainingSet(21, 4) \
    .import_from_file('car_evaluation.txt', ',') \
    .slice(80)

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
                                                   max_error=0.001)

# Configuring a log after each learning method iteration
momentum_backpropagation.on_after_iteration = lambda b: print(b.actual_iteration, ':', b.total_network_error)

# Learning the training_set
momentum_backpropagation.learn(training_set_training)

# Printing results
right_classification = 0
wrong_classification = 0
discarded_classification = 0

for training_set_row in training_set_test:

    output = multi_layer_perceptron \
        .set_input(training_set_row.input_pattern) \
        .compute() \
        .output

    for i in range(len(output)):
        if (0 - sim) <= output[i] <= (0 + sim):
            output[i] = 0
        elif (1 - sim) <= output[i] <= (1 + sim):
            output[i] = 1
        else:
            output[i] = None

    if output is None:
        discarded_classification += 1
    elif output == training_set_row.ideal_output:
        right_classification += 1
    else:
        wrong_classification += 1

Logger.info('right_classification = ', right_classification, ' ',
            round((100 / len(training_set_test)) * right_classification, 2),
            '%')
Logger.info('wrong_classification = ', wrong_classification, ' ',
            round((100 / len(training_set_test)) * wrong_classification, 2),
            '%')
Logger.info('discarded_classification = ', discarded_classification, ' ',
            round((100 / len(training_set_test)) * discarded_classification, 2), '%')