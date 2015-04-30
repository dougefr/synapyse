from synapyse.base.learning.training_set import TrainingSet

__author__ = 'Douglas Eric Fonseca Rodrigues'

# Define o sim da aplicação
sim = 0.5

# Importa os dados para treinamento
dados_treinamento, dados_teste = TrainingSet(input_count=4, output_count=3) \
    .import_from_file('iris.data') \
    .normalize() \
    .shuffle() \
    .slice(70)

dados_treinamento.save_to_file('iris_training.data')
dados_teste.save_to_file('iris_testing.data')

