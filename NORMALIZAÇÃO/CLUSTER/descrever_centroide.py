import pickle
import pandas as pd

#matriz de nomes das colunas do arquivo csv (antes de treinar os clusters)
nomesColunas = ['sepal_length', 
                'sepal_width', 
                'petal_length', 
                'petal_width',
                'Iris-setosa', 
                'Iris-versicolor', 
                'Iris-virginica']

#abrir modelo treinado(custer)
clusterIris = pickle.load(open('clusterIris.pkl', 'rb'))

'''
#imprimir os valores dos centroides
print(clusterIris.cluster_centers_)

'''

#converte centroides em dataframes
centroides = pd.DataFrame(clusterIris.cluster_centers_, columns = nomesColunas)

#separar o dataframe em colunas numericas e categoricas
dados_num_norm = centroides.drop(columns=['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'])

dados_cat_norm = centroides[['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']]

#desnormalizar as colunas numericas 
#carregar normalizador salvo durante processsamento
normalizador = pickle.load(open('normalizadorIris.pkl', 'rb'))
dadosNum = normalizador.inverse_transform(dados_num_norm)

#APÓS DESNORMALIZAR DADOS NUMERICOS TERÁ UMA MATRIZ DO NUMPY
#PRECISA RECRIAR O DATAFRAME
dadosNum = pd.DataFrame(dadosNum, columns = dados_num_norm.columns)


#desnormalizar colunas categoricas
dadosCat = pd.from_dummies(dados_cat_norm.round(0).astype(int))

dadosCat.columns = ['Class']
#print(dadosCat)

#junta os 2 dataFrames
clusterIris_dados = dadosNum.join(dadosCat)

print(clusterIris_dados)












