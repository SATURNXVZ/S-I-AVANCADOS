import pandas as pd
import numpy as np
import pickle
import os


def main():
    print("="*60)
    print("ANÁLISE DOS CLUSTERS")
    print("="*60)
    
    COLUNAS_REMOVER = ['NObeyesdad_Insufficient_Weight', 'NObeyesdad_Normal_Weight', 
                   'NObeyesdad_Obesity_Type_I', 'NObeyesdad_Obesity_Type_II',
                   'NObeyesdad_Obesity_Type_III', 'NObeyesdad_Overweight_Level_I',
                   'NObeyesdad_Overweight_Level_II']

    
    #carregar modelo KMeans
    nome_modelo = "modelo_kmeans.pkl"
    
    if not os.path.exists(nome_modelo):
        print(f"\nERRO! Modelo não encontrado: {nome_modelo}")
        return
    
    with open(nome_modelo, 'rb') as f:
        kmeans = pickle.load(f)
    
    print(f"Modelo carregado: {nome_modelo}")
    
    dados_processados = None
    
    #carrega csv normalizado (ALTERAR)
    caminho_proc = r"C:\Users\VITORHENRIQUEDEMELO\Documents\S-I-AVANCADOS\PROVA\dados_normalizar_processado.csv"
    if os.path.exists(caminho_proc):
            dados_processados = pd.read_csv(caminho_proc)
            dados_processados = dados_processados.drop(columns=[c for c in COLUNAS_REMOVER if c in dados_processados.columns])
            print(f"Dados carregados: {dados_processados.shape}")
    else:
        print(f"ERRO! Arquivo não encontrado: {caminho_proc}")
    
    #mostrar centroides normalizados
    print("\nCENTROIDES (dados normalizados):")
    centroides_norm = pd.DataFrame(kmeans.cluster_centers_)
    print(centroides_norm)
    
    #carregar normalizador e desnormalizar automaticamente
    from minmaxscaler import MinMaxScalerProcessor
    caminho_scaler = r"C:\Users\VITORHENRIQUEDEMELO\Documents\S-I-AVANCADOS\PROVA\processadores\min_max_scaler.pkl"
    
    if os.path.exists(caminho_scaler):
        scaler = MinMaxScalerProcessor.load(caminho_scaler)
        print(f"Normalizador carregado!")
        
        if hasattr(scaler, 'column_names') and scaler.column_names:
            colunas_scaler = scaler.column_names
            print(f"Colunas do normalizador: {colunas_scaler}")
            
            num_colunas = min(len(colunas_scaler), centroides_norm.shape[1])
            colunas_para_usar = colunas_scaler[:num_colunas]
            
            centroides_df = pd.DataFrame(
                centroides_norm.iloc[:, :num_colunas].values,
                columns=colunas_para_usar
            )
            
            print(f"\nCentroides com nomes de colunas:")
            print(centroides_df)
            
            try:
                centroides_reais = scaler.inverse_transform(centroides_df)
                print("\nCENTROIDES (VALORES REAIS - DESNORMALIZADOS):")
                print(centroides_reais)
                
            except Exception as e:
                print(f"Erro ao desnormalizar: {e}")
                print("\nTentando outra forma")
                try:
                    dados_reais_array = scaler.scaler.inverse_transform(centroides_df.values)
                    df_reais = pd.DataFrame(dados_reais_array, columns=colunas_para_usar)
                    print("\nCENTROIDES (VALORES REAIS - MÉTODO ALTERNATIVO):")
                    print(df_reais)
                except Exception as e2:
                    print(f"ERRO! Também falhou: {e2}")
        else:
            print("\nNormalizador nao tem nomes de colunas")
    else:
        print(f"ERRO! Normalizador não encontrado: {caminho_scaler}")
    
    #informações gerais
    print("\nINFORMAÇÕES DOS CLUSTERS:")
    print(f"  Número de clusters: {kmeans.n_clusters}")
    print(f"  Inércia: {kmeans.inertia_:.2f}") #.2f pára 2 casa decimais após virgula
    
    if hasattr(kmeans, 'n_iter_'):
        print(f"  Iterações: {kmeans.n_iter_}")
    
    #mostrar distribuição (se tiver os dados)
    if dados_processados is not None:
        labels = kmeans.predict(dados_processados)
        print("\nDISTRIBUICAO DOS CLUSTERS:")
        for i in range(kmeans.n_clusters):
            count = np.sum(labels == i)
            print(f"  Cluster {i}: {count} pontos ({count/len(labels)*100:.1f}%)")
    
    print("\nANÁLISE FINALIZADA!")


if __name__ == "__main__":
    main()