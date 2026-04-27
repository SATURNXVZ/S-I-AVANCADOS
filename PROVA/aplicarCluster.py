import pickle
import pandas as pd
import numpy as np
import os

from minmaxscaler import MinMaxScalerProcessor
from onehot import OneHotEncoderProcessor

CAMINHO_MODELO     = r"C:\Users\VITORHENRIQUEDEMELO\Documents\S-I-AVANCADOS\PROVA\modelo_kmeans.pkl"
CAMINHO_COLUNAS    = r"C:\Users\VITORHENRIQUEDEMELO\Documents\S-I-AVANCADOS\PROVA\colunas_cluster.pkl"
PASTA_PROCESSADORES = r"C:\Users\VITORHENRIQUEDEMELO\Documents\S-I-AVANCADOS\PROVA\processadores"


#DADOS DE NOVA INFERENCIA
nova_instancia = pd.DataFrame([[
    'Female',       # Gender
    25,             # Age
    1.65,           # Height
    70,             # Weight
    'yes',          # family_history_with_overweight
    'yes',          # FAVC
    2.0,            # FCVC
    3.0,            # NCP
    'Sometimes',    # CAEC
    'no',           # SMOKE
    2.0,            # CH2O
    'no',           # SCC
    1.0,            # FAF
    1.0,            # TUE
    'no',           # CALC
    'Public_Transportation'  # MTRANS
]], columns=[
    'Gender', 'Age', 'Height', 'Weight', 'family_history_with_overweight',
    'FAVC', 'FCVC', 'NCP', 'CAEC', 'SMOKE', 'CH2O', 'SCC', 'FAF',
    'TUE', 'CALC', 'MTRANS'
])

print(f"Carregando modelo: {CAMINHO_MODELO}")
with open(CAMINHO_MODELO, 'rb') as f:
    kmeans = pickle.load(f)
print(f"  Modelo carregado, Clusters: {kmeans.n_clusters}")

with open(CAMINHO_COLUNAS, 'rb') as f:
    colunas = pickle.load(f)
print(f"  Colunas carregadas: {len(colunas)} features")

#carregar processadores
scaler = MinMaxScalerProcessor.load(f"{PASTA_PROCESSADORES}/min_max_scaler.pkl")

colunas_onehot = [
    'Gender', 'family_history_with_overweight', 'FAVC',
    'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS'
]

#aplicar onehot nas colunas categoricas
df = nova_instancia.copy()
for col in colunas_onehot:
    if col in df.columns:
        enc = OneHotEncoderProcessor.load(f"{PASTA_PROCESSADORES}/one_hot_encoder_{col}.pkl")
        encoded = enc.transform(df)
        df = df.drop(col, axis=1)
        df = pd.concat([df, encoded], axis=1)

#aplicar MinMax nas colunas numericas
colunas_num = scaler.column_names
df[colunas_num] = scaler.scaler.transform(df[colunas_num])

#alinhar colunas com o modelo (ordem e colunas ausentes)
for col in colunas:
    if col not in df.columns:
        df[col] = 0
df = df[colunas]

print(f"\nNova instância preparada:")
print(df)

#print(df.to_string()) testes.

#predizer cluster
cluster = kmeans.predict(df.values)
print(f'\nCluster previsto: {cluster[0]}')