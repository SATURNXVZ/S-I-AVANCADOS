import pickle as pk
import pandas as pd
import sklearn as sk
import numpy as np

#o pandas tem um metodo concat()
#criar dataframe 
florNorm = pd.DataFrame(columns = ['sepal_length', 
                'sepal_width', 
                'petal_length', 
                'petal_width',
                'Iris-setosa', 
                'Iris-versicolor', 
                'Iris-virginica'])

#no sisterma final esses dados seráo recebidos
novaFlor = pd.DataFrame([[6.4, 2.8, 5.6, 2.1]], columns = ['sepal_length', 
                                                        'sepal_width', 
                                                        'petal_length', 
                                                        'petal_width',] )

#normalizar nvoa flor
#carregar normalizador salvo durante treinamento
normalizador = pk.load(open('normalizadorIris.pkl', 'rb'))
novaFlor = normalizador.transform(novaFlor)

novaFlorNorm = pd.DataFrame(novaFlor, columns = ['sepal_length', 
                                                        'sepal_width', 
                                                        'petal_length', 
                                                        'petal_width',] )

#inferir cluster a qual a flor o pertence
#carregar modelo de cluster

clusterIris = pk.load(open('clusterIris.pkl', 'rb'))
clusterNovaFlor = clusterIris.predict(novaFlor)
print(clusterNovaFlor)

#criar 



