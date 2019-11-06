import pandas as pd
from sklearn.utils import optimize

previsores = pd.read_csv("entradas-breast.csv")
clase = pd.read_csv("saidas-breast.csv")


from sklearn.model_selection import train_test_split
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, clase, test_size=0.25)


import keras
from keras.models import Sequential
from keras.layers import Dense

# a rede em se
classificador = Sequential
'''
units= 16 = neuronios da camada oculta = entradas + saidas / 2
activation='relu' = função de ativação, por hora usar essa
kernel_initializer='random_uniform' = inicialização dos pessos aleatórios
input_dim=30 = número de entradas
'''
classificador.add(Dense(units=16, activation='relu', kernel_initializer='random_uniform', input_dim=30))
classificador.add(Dense(units=1, activation='sigmoid'))

#classificador.compile(optimize='adam', loss='binary_crossentropy')