from impl.activation_functions.sigmoid import Sigmoid
from impl.input_functions.weighted_sum import WeightedSum
from impl.learning.momentum_back_propagation import MomentumBackPropagation
from impl.multi_layer_perceptron import MultiLayerPerceptron
from util.training_set_util import import_from_file

__author__ = 'Douglas Eric Fonseca Rodrigues'


def test(multi_layer_perceptron, training_set):
    for training_set_row in training_set:
        print("Input: ", training_set_row.input_pattern)
        print("Ideal output\t: ", training_set_row.ideal_output)

        output = multi_layer_perceptron \
            .set_input(training_set_row.input_pattern).compute() \
            .output

        print("Resulted output\t: ", output)


def main():
    training_set = import_from_file("/home/douglas/projects/synapyse/test/impl/car_evaluation.txt", 21, 4, ",")

    multi_layer_perceptron = MultiLayerPerceptron()

    multi_layer_perceptron \
        .create_layer(21, WeightedSum) \
        .create_layer(14, WeightedSum, Sigmoid) \
        .create_layer(4, WeightedSum, Sigmoid) \
        .randomize_weights()

    print(multi_layer_perceptron.weights)

    momentum_backpropagation = MomentumBackPropagation(neural_network=multi_layer_perceptron,
                                                       learning_rate=0.3,
                                                       momentum=0.6,
                                                       max_error=0.01)

    momentum_backpropagation.on_after_iteration = lambda b: print(b.actual_iteration, ":", b.total_network_error)

    momentum_backpropagation.learn(training_set)

    print(multi_layer_perceptron.weights)

    # test(multi_layer_perceptron, training_set)


main()
# profile.run("main(); print")
