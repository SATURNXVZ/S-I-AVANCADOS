import pandas as pd
from sklearn.cluster import KMeans #metaestimador
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import cdist
import numpy as np
import pickle as pk
import matplotlib.pyplot as plt
import math 


#abrir csv

dados = pd.read_csv('NORMALIZAÇÃO\CLUSTER\iris.csv', sep=';')

#print(dados)

#separar atributos numéricos dos categóricos
num = dados.drop(columns=['class'])
cat = dados['class']

#print(cat.head(10))
#print(num.head(10))

#normalizar numerico
scaler = MinMaxScaler()
normalizador = scaler.fit(num)

#salvar modelo normalizado
pk.dump(normalizador, open('normalizadorIris.pkl', 'wb'))
numNorm = normalizador.fit_transform(num)

#salvar 
catNorm = pd.get_dummies(cat, prefix_sep= '_', dtype=int)
#print(catNorm.head(10))

#numeros normalizados perdeu os nomes das colunas
#temos que deixar com rótulo
print(num.columns)
#com isso em mente, criar um dataframe com os normalizados
#usando o nome das colunas antigas

#transformar o numNorm em dataFrame
numNorm = pd.DataFrame(numNorm, columns = num.columns)

#recriar dataFrame com todos os dados
dadosNorm = numNorm.join(catNorm, how='left')

#print(dadosNorm.head(10))

#hiperparametro antes de treinar
distorcoes = []

K = range(1, 101)

#treinando iterativamente, aumentando o numero de clusters
for i in K:
    clusterIris = KMeans(n_clusters=i, random_state=42).fit(dadosNorm)
    
    #calcular distorcao
    distorcoes.append(
        sum(
            np.min(
                cdist(
                    dadosNorm, clusterIris.cluster_centers_, 
                    'euclidean'), axis=1)/dadosNorm.shape[0]
            )
    )
    
print(distorcoes);  
    