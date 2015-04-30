from synapyse.base.learning.training_set import TrainingSet
from synapyse.impl.learning.least_mean_square import LeastMeanSquare
from synapyse.impl.perceptron import Perceptron

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
for taxa_aprendizado in [0.01, 0.1, 0.3, 0.5, 0.7, 0.9]:

    # Cria a rede neural artificial
    rede_neural = Perceptron(input_count=4, output_count=3) \
        .randomize_weights()

    # Cria o algoritmo de aprendizado
    aprendizado = LeastMeanSquare(neural_network=rede_neural,
                                  learning_rate=taxa_aprendizado,
                                  max_error=0.01,
                                  max_iterations=5000)

    print('Treinamento iniciado com taxa de aprendizado =', taxa_aprendizado)

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
            print(dado.input_pattern, dado.ideal_output, output)
            classificacoes_incorretas += 1
        else:
            classificacoes_descartadas += 1

    print('Número de iterações:', aprendizado.actual_iteration)
    print('Classificações corretas:', classificacoes_corretas,
          (100 / len(dados_teste)) * classificacoes_corretas)
    print('Classificações incorretas:', classificacoes_incorretas,
          (100 / len(dados_teste)) * classificacoes_incorretas)
    print('Classificações descartadas:', classificacoes_descartadas,
          (100 / len(dados_teste)) * classificacoes_descartadas)