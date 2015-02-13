import json

import jsonpickle

from synapyse.base.learning.training_set import TrainingSet
from synapyse.impl.activation_functions.tanh import Tanh
from synapyse.impl.input_functions.weighted_sum import WeightedSum
from synapyse.impl.learning.back_propagation import BackPropagation
from synapyse.impl.multi_layer_perceptron import MultiLayerPerceptron


__author__ = 'Douglas Eric Fonseca Rodrigues'

training_set = TrainingSet(2, 1)

training_set.append([1.0, 1.0], [1.0])

n = MultiLayerPerceptron()

n \
    .create_layer(2, WeightedSum()) \
    .create_layer(3, WeightedSum(), Tanh(2)) \
    .create_layer(1, WeightedSum(), Tanh(2))

n00 = n.layers[0].neurons[0]
n01 = n.layers[0].neurons[1]
b0 = n.layers[0].neurons[2]

n10 = n.layers[1].neurons[0]
n11 = n.layers[1].neurons[1]
n12 = n.layers[1].neurons[2]
b1 = n.layers[1].neurons[3]

n20 = n.layers[2].neurons[0]

# Layer 02
n10.input_connections[n00].weight = 0.029909199228032972
n10.input_connections[n01].weight = -0.05335685075082475
n10.input_connections[b0].weight = -0.13007778657509603

n11.input_connections[n00].weight = -0.7848359490460989
n11.input_connections[n01].weight = -0.20753519050536662
n11.input_connections[b0].weight = 0.32799030664796486

n12.input_connections[n00].weight = 0.26618074588708085
n12.input_connections[n01].weight = 0.7266372719185705
n12.input_connections[b0].weight = -0.7709852516518709

# Layer 04
n20.input_connections[n10].weight = 0.6749469319347992
n20.input_connections[n11].weight = 0.787375982611922
n20.input_connections[n12].weight = -0.5524561873998115
n20.input_connections[b1].weight = -0.7512327441111853

# print(json.dumps(json.loads(jsonpickle.encode(multi_layer_perceptron)), indent=4, sort_keys=True))

b = BackPropagation(n, learning_rate=0.1, max_iterations=1, max_error=0.1)

b.learn(training_set)

print(json.dumps(json.loads(jsonpickle.encode(n)), indent=4, sort_keys=True))

print(n.weights)
