import pickle
import pandas as pd
import numpy as np
import os


CAMINHO_MODELO = "modelo_kmeans.pkl"
CAMINHO_COLUNAS = "colunas_cluster.pkl"  #precisa salvar no treino


#verifica se o modelo existe
if not os.path.exists(CAMINHO_MODELO):
    print(f"\nERRO! Modelo não encontrado: {CAMINHO_MODELO}")
    exit()

#carregar modelo
print(f"📂 Carregando modelo: {CAMINHO_MODELO}")
with open(CAMINHO_MODELO, 'rb') as f:
    kmeans = pickle.load(f)
print(f" Modelo carregado, Clusters: {kmeans.n_clusters}")

#carregar colunas do treino
if os.path.exists(CAMINHO_COLUNAS):
    with open(CAMINHO_COLUNAS, 'rb') as f:
        colunas = pickle.load(f)
    print(f"Colunas carregadas: {len(colunas)} features")
else:
    print(f" ERRO! Arquivo de colunas não encontrado: {CAMINHO_COLUNAS}")
    print(f" ERRRO! O modelo espera {kmeans.n_features_in_} features")
    print(f"ERRO! Execute o treinamento novamente para salvar as colunas")
    exit()

#criar nova instância com ZERO em todas as colunas
nova_instancia = pd.DataFrame(0, index=[0], columns=colunas)

"""
VALORES DA SUA NOVA INSTANCIA
exemplo com valores normalizados (0 e 1)

nova_instancia[''] = 0
============================================================

print(f"\nNova instancia preparada:")
print(nova_instancia)
"""

#predizer cluster
cluster = kmeans.predict(nova_instancia)
print(f'\ncluster previsto: {cluster[0]}')