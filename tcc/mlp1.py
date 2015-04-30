from synapyse.base.learning.training_set import TrainingSet
from synapyse.impl.activation_functions.sigmoid import Sigmoid
from synapyse.impl.input_functions.weighted_sum import WeightedSum
from synapyse.impl.learning.momentum_back_propagation \
    import MomentumBackPropagation
from synapyse.impl.multi_layer_perceptron import MultiLayerPerceptron

__author__ = 'Douglas Eric Fonseca Rodrigues'

# Define o sim da aplicação
sim = 0.5

# Importa os dados para treinamento
dados_treinamento = TrainingSet(input_count=4, output_count=3) \
    .import_from_file('iris_training.data')

# Importa os dados para teste
dados_teste = TrainingSet(input_count=4, output_count=3) \
    .import_from_file('iris_testing.data')

# Testa diversas taxas de aprendizagem
for numero_neuros_camada_oculta in [1, 2, 3, 4, 5, 6, 7]:

    # Cria a rede neural artificial
    rede_neural = MultiLayerPerceptron() \
        .create_layer(neuron_count=4,
                      input_function=WeightedSum()) \
        .create_layer(neuron_count=numero_neuros_camada_oculta,
                      input_function=WeightedSum(),
                      activation_function=Sigmoid()) \
        .create_layer(neuron_count=3,
                      input_function=WeightedSum(),
                      activation_function=Sigmoid()) \
        .randomize_weights()

    # Cria o algoritmo de aprendizado
    aprendizado = MomentumBackPropagation(neural_network=rede_neural,
                                          learning_rate=0.7,
                                          max_error=0.05,
                                          momentum=0,
                                          max_iterations=30)

    print('Rede com número de neurônios na camada oculta =',
          numero_neuros_camada_oculta)

    # Inicia o aprendizado
    aprendizado.learn(dados_treinamento)

    # Testa o aprendizado
    classificacoes_corretas = 0
    classificacoes_incorretas = 0
    classificacoes_descartadas = 0

    possiveis_classificacoes = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

    for dado in dados_teste:
        output = rede_neural.set_input(dado.input_pattern) \
            .compute() \
            .output

        # Ajusta output para verificação do resultado
        for i in range(len(output)):
            if (1 - sim) < output[i] < (1 + sim):
                output[i] = 1
            elif (0 - sim) < output[i] < (0 + sim):
                output[i] = 0
            else:
                output[i] = None

        # Verifica a classificação
        if output == dado.ideal_output:
            classificacoes_corretas += 1
        elif output in possiveis_classificacoes:
            classificacoes_incorretas += 1
        else:
            classificacoes_descartadas += 1

    print('Número de iterações:', aprendizado.actual_iteration)
    print('Classificações corretas:', classificacoes_corretas, (100 / len(dados_teste)) * classificacoes_corretas)
    print('Classificações incorretas:', classificacoes_incorretas, (100 / len(dados_teste)) * classificacoes_incorretas)
    print('Classificações descartadas:', classificacoes_descartadas,
          (100 / len(dados_teste)) * classificacoes_descartadas)