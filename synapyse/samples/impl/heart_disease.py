import xlsxwriter

from synapyse.impl.learning.back_propagation import BackPropagation
from synapyse.base.learning.training_set import TrainingSet
from synapyse.impl.activation_functions.sigmoid import Sigmoid
from synapyse.impl.input_functions.weighted_sum import WeightedSum
from synapyse.impl.multi_layer_perceptron import MultiLayerPerceptron


__author__ = 'Douglas Eric Fonseca Rodrigues'

# Creating a training_set based in a text file
training_set = TrainingSet(13, 1) \
    .import_from_file('heart_disease.txt', ',') \
    .normalize()

# Creating and configuring the network
multi_layer_perceptron = MultiLayerPerceptron() \
    .create_layer(13, WeightedSum()) \
    .create_layer(8, WeightedSum(), Sigmoid()) \
    .create_layer(1, WeightedSum(), Sigmoid()) \
    .randomize_weights()

# Generating an excel-file result
# Create a workbook and add a worksheet.
workbook = xlsxwriter.Workbook('heart_disease.xlsx')

for learning_rate in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:

    print('Running learning_rate =', learning_rate, '...')

    # Creating and configuring the learning method
    momentum_backpropagation = BackPropagation(neural_network=multi_layer_perceptron,
                                               learning_rate=learning_rate,
                                               max_error=0.01,
                                               max_iterations=5000)

    # Configuring a log after each learning method iteration
    errors = []
    momentum_backpropagation.on_after_iteration = lambda b: (
        errors.append([b.actual_iteration, b.total_network_error])
    )

    # Learning the training_set
    momentum_backpropagation.learn(training_set)

    worksheet_name = 'testing learning_rate=' + str(learning_rate)
    worksheet = workbook.add_worksheet(worksheet_name)

    row = 0
    col = 0

    for iteration, error in errors:
        worksheet.write(row, col, error)
        row += 1

    # Create a new chart object.
    chart = workbook.add_chart({'type': 'line'})
    chart.add_series({'values': ('=' + worksheet_name + '!$A$1:$A$' + str(len(errors)))})
    worksheet.insert_chart('C1', chart)

workbook.close()
