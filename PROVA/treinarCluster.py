import pandas as pd
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import math
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans

# ============================================================
# CONFIGURAÇÕES (você ajusta aqui antes de rodar)
# ============================================================
caminho_dados = r"C:\Users\Pichau\OneDrive\Documentos\Code\S-I-AVANCADOS\PROVA\dados_normalizar_processado.csv"
COLUNAS_REMOVER = [""]  #remova colunas muito espeficias, deixe vazio se não quiser remover nada
MAX_K = 5 #numero máximo de clusters para testar, adiicone menos para base de dados menor
NOME_MODELO = "modelo_kmeans.pkl"
# ============================================================

#melhor numero de cluster usando cotovelo
def encontrar_melhor_k(dados, max_k=20):
    #Garante que max_k não ultrapassa o numero de amostras
    max_k = min(max_k, len(dados) - 1)
    distorcoes = []
    K_range = range(1, max_k + 1)
    
    print(f"\nTestando clusters de 1 a {max_k}...")
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(dados)
        
        distorcao = sum(
            np.min(cdist(dados, kmeans.cluster_centers_, 'euclidean'), axis=1)
        ) / dados.shape[0]
        distorcoes.append(distorcao)
    
    #metodo geometrico para encontrar o "cotovelo"
    x0, y0 = K_range[0], distorcoes[0]
    xn, yn = K_range[-1], distorcoes[-1]
    
    distancias = []
    for i in range(len(distorcoes)):
        x, y = K_range[i], distorcoes[i]
        numerador = abs((yn - y0) * x - (xn - x0) * y + xn * y0 - yn * x0)
        denominador = math.sqrt((yn - y0)**2 + (xn - x0)**2)
        distancias.append(numerador / denominador)
    
    k_otimo = K_range[distancias.index(max(distancias))]
    
    print(f"\nMelhor número de clusters: {k_otimo}")    
    return k_otimo

def main():
    print("="*60)
    print("TREINAMENTO DE CLUSTERIZAÇÃO")
    print("="*60)
    
    #carregar dados
    print(f"\nCarregando dados: {caminho_dados}")
    dados = pd.read_csv(caminho_dados)
    print(f"  {dados.shape[0]} linhas x {dados.shape[1]} colunas")
    
    print(f"\nColunas disponiveis: {dados.columns.tolist()}")
    
    #remover colunas configuradas
    if COLUNAS_REMOVER:
        colunas_remover = [col for col in COLUNAS_REMOVER if col in dados.columns]
        if colunas_remover:
            dados = dados.drop(columns=colunas_remover)
            print(f"Removidas: {colunas_remover}")
    
    print(f"\nDados para clusterização: {dados.shape[0]} linhas x {dados.shape[1]} colunas")
    
    #econtrar melhor k
    k_otimo = encontrar_melhor_k(dados.values, max_k=MAX_K)
    
    #treinar modelo final
    print(f"\nTreinando K-Means com {k_otimo} clusters...")
    kmeans = KMeans(n_clusters=k_otimo, random_state=42, n_init=10)
    kmeans.fit(dados.values)
    
    #Depois de treinar o modelo (antes ou depois de salvar o kmeans)
    with open('colunas_cluster.pkl', 'wb') as f:
        pickle.dump(dados.columns.tolist(), f)
    print(f"Colunas salvas: colunas_cluster.pkl")
    
    #salvar modelo
    with open(NOME_MODELO, 'wb') as f:
        pickle.dump(kmeans, f)
    print(f"Modelo salvo: {NOME_MODELO}")
    
    #mostrar resultados
    print("\n" + "="*60)
    print("RESULTADOS")
    print("="*60)
    
    labels = kmeans.labels_
    print("\nDistribuição dos clusters:")
    for i in range(k_otimo):
        count = np.sum(labels == i)
        print(f"  Cluster {i}: {count} pontos ({count/len(labels)*100:.1f}%)")
    
    print(f"\n  Inercia total: {kmeans.inertia_:.2f}")
    print(f"    Numero de iteracoes: {kmeans.n_iter_}")
    
    print("\n   TREINAMENTO FINALIZADO!")


if __name__ == "__main__":
    main()