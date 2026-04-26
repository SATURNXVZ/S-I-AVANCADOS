import pandas as pd
import numpy as np
import os 
import warnings
warnings.filterwarnings('ignore')

#importa os modos de normalizacao
from onehot import OneHotEncoderProcessor
from minmaxscaler import MinMaxScalerProcessor
from label import LabelEncoderProcessor

#trata valores nulos (NA) no dataframe
def tratar_nulos(df):
        if df.isnull().sum().sum() == 0:
            print("Nenhum valor nulo detectado")
            return df
        
        print("\nValores nulos detectados:")
        print(df.isnull().sum())
        
        #remove linhas com nulos
        df_limpo = df.dropna()
        
        print(f"Linhas removidas: {len(df) - len(df_limpo)}")
        print(f"Linhas restantes: {len(df_limpo)}")
        
        return df_limpo

class ProcessadorDadosCompleto:
    def __init__(self):
        self.one_hot_encoders = {}  #por coluna, evita sobrescrever o estado
        self.label_encoder = LabelEncoderProcessor()
        self.scaler = MinMaxScalerProcessor()
        self.processed_columns = {}
        self.numeric_columns = []
        self.one_hot_columns = []
        self.label_columns = []
    
    #descobre qual tipo de dado a coluna tem    
    def identificar_tipos_colunas(self, df):
        #colunas numéricas
        self.numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        #colunas categóricas (object, category)
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        #separar colunas para OneHot (nominal) e label (ordinais)
        self.one_hot_columns = []
        self.label_columns = []
        
        for col in categorical_cols:
            unique_count = df[col].nunique()
            if unique_count <= 10:  #se poucas categorias, usa one-hot
                self.one_hot_columns.append(col)
            else:  #se muitas, usa label
                self.label_columns.append(col)
        
        return {
            'numeric': self.numeric_columns,
            'one_hot': self.one_hot_columns,
            'label': self.label_columns
        }
    
    #aplica onehot nas colunas identificadas antes
    def aplicar_one_hot_encoding(self, df):
        if not self.one_hot_columns:
            return df
        
        df_resultado = df.copy()
        
        for col in self.one_hot_columns:
            if col in df_resultado.columns:
                print(f"One-Hot: {col} → {df[col].nunique()} categorias")
                enc = OneHotEncoderProcessor()          #encoder independente por coluna
                self.one_hot_encoders[col] = enc
                encoded_data = enc.fit_transform(df_resultado, col)
                df_resultado = df_resultado.drop(col, axis=1)
                df_resultado = pd.concat([df_resultado, encoded_data], axis=1)
        
        return df_resultado
    
    #transforma categorias em numeros
    def aplicar_label_encoding(self, df):
        if not self.label_columns:
            return df
        
        print(f"  Label: {self.label_columns}")
        df_resultado = self.label_encoder.fit_transform(df, self.label_columns)
        return df_resultado
    
    #aplicar minmax (colunas numericas)
    def aplicar_min_max_scaler(self, df):
        if not self.numeric_columns:
            return df
        
        print(f"  Scaler: {self.numeric_columns}")
        colunas_para_escalar = [c for c in self.numeric_columns if c in df.columns]  #só numéricas originais
        df_resultado = self.scaler.fit_transform(df, columns=colunas_para_escalar)
        return df_resultado
    
    #processa df completo com as informações "normalizadas"
    def processar_completo(self, df):
        print("="*60)
        print("INICIANDO PROCESSAMENTO")
        print("="*60)
        
        #identificar tipos de colunas
        tipos = self.identificar_tipos_colunas(df)
        
        print(f"\n DADOS ORIGINAIS:")
        print(f"  Linhas: {len(df)}")
        print(f"  Colunas: {len(df.columns)}")
        print(f"  Numéricas: {len(tipos['numeric'])}")
        print(f"  One-Hot (≤10 cat): {len(tipos['one_hot'])}")
        print(f"  Label (>10 cat): {len(tipos['label'])}")
        
        df_processado = df.copy()
        
        df_processado = self.aplicar_one_hot_encoding(df_processado)
        df_processado = self.aplicar_label_encoding(df_processado)
        df_processado = self.aplicar_min_max_scaler(df_processado)
        
        print(f"\n RESULTADO FINAL:")
        print(f"  Colunas originais: {len(df.columns)}")
        print(f"  Colunas finais: {len(df_processado.columns)}")
        print("="*60)
        
        return df_processado
    
    #salva os pkls treinados
    def salvar_processadores(self, caminho_base):
        os.makedirs(caminho_base, exist_ok=True)
        for col, enc in self.one_hot_encoders.items():  #salva cada encoder
            enc.save(f"{caminho_base}/one_hot_encoder_{col}.pkl")
        self.label_encoder.save(f"{caminho_base}/label_encoder.pkl")
        self.scaler.save(f"{caminho_base}/min_max_scaler.pkl")
        print(f"\nProcessadores salvos em: {caminho_base}")


def main():
    print("="*60)
    print("SISTEMA DE PRÉ-PROCESSAMENTO DE DADOS")
    print("="*60)
    
    #caminho da base de dados
    caminho_arquivo = r"C:\Users\Pichau\OneDrive\Documentos\Code\S-I-AVANCADOS\PROVA\dados_normalizar.csv"
    
    #ler arquivo CSV
    print("\nCarregando arquivo...")
    df_original = pd.read_csv(caminho_arquivo, encoding='utf-8')
    print(f"  {df_original.shape[0]} linhas x {df_original.shape[1]} colunas")
    
    #mostrar amostra
    print("\n📋 AMOSTRA DOS DADOS ORIGINAIS:")
    print(df_original.head())
    
    #processar
    df_original = tratar_nulos(df_original)
    processador = ProcessadorDadosCompleto()
    df_processado = processador.processar_completo(df_original)
    
    #mostrar resultado
    print("\n📋 AMOSTRA DOS DADOS PROCESSADOS:")
    print(df_processado.head())
    
    #salvar
    caminho_saida = caminho_arquivo.replace('.csv', '_processado.csv')
    df_processado.to_csv(caminho_saida, index=False)
    print(f"\n💾 Dados salvos em: {caminho_saida}")
    
    #salvar processadores
    caminho_processadores = "PROVA/processadores"
    processador.salvar_processadores(caminho_processadores)


if __name__ == "__main__":
    main()