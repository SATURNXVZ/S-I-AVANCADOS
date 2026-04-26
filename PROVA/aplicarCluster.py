import pickle
import pandas as pd
import numpy as np
import os

# ============================================================
# CONFIGURAÇÕES
# ============================================================
CAMINHO_MODELO = "modelo_kmeans.pkl"
CAMINHO_COLUNAS = "colunas_cluster.pkl"  # você precisa salvar no treino
# ============================================================

# Verificar se o modelo existe
if not os.path.exists(CAMINHO_MODELO):
    print(f"❌ Modelo não encontrado: {CAMINHO_MODELO}")
    exit()

# Carregar modelo
print(f"📂 Carregando modelo: {CAMINHO_MODELO}")
with open(CAMINHO_MODELO, 'rb') as f:
    kmeans = pickle.load(f)
print(f"  ✅ Modelo carregado | Clusters: {kmeans.n_clusters}")

# Carregar colunas do treino
if os.path.exists(CAMINHO_COLUNAS):
    with open(CAMINHO_COLUNAS, 'rb') as f:
        colunas = pickle.load(f)
    print(f"  ✅ Colunas carregadas: {len(colunas)} features")
else:
    print(f"  ⚠️ Arquivo de colunas não encontrado: {CAMINHO_COLUNAS}")
    print(f"  O modelo espera {kmeans.n_features_in_} features")
    print(f"  Execute o treinamento novamente para salvar as colunas")
    exit()

# Criar nova instância com ZERO em todas as colunas
nova_instancia = pd.DataFrame(0, index=[0], columns=colunas)

"""============================================================
PREENCHA AQUI OS VALORES DA SUA NOVA INSTANCIA
Exemplo com valores normalizados (entre 0 e 1)

nova_instancia[''] = 0
============================================================

print(f"\n📋 Nova instância preparada:")
print(nova_instancia)
"""

# Predizer cluster
cluster = kmeans.predict(nova_instancia)
print(f'\n🎯 Cluster previsto: {cluster[0]}')