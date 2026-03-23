import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pickle

class MinMaxScalerProcessor:
    def __init__(self, feature_range=(0, 1)):
        """
        Inicializa o MinMaxScaler
        
        Args:
            feature_range: Tupla (min, max) definindo o intervalo desejado
        """
        self.scaler = MinMaxScaler(feature_range=feature_range)
        self.column_names = None
    
    def fit(self, df, columns=None):
        """
        Aplica fit no MinMaxScaler
        
        Args:
            df: DataFrame com os dados
            columns: Lista de colunas para escalonar (se None, usa todas colunas numéricas)
        """
        if columns is None:
            # Seleciona apenas colunas numéricas
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        self.column_names = columns
        self.scaler.fit(df[columns])
        return self
    
    def transform(self, df):
        """
        Transforma os dados aplicando MinMaxScaler
        
        Args:
            df: DataFrame com os dados
        
        Returns:
            DataFrame com os dados escalonados
        """
        df_scaled = df.copy()
        scaled_data = self.scaler.transform(df[self.column_names])
        df_scaled[self.column_names] = scaled_data
        return df_scaled
    
    def fit_transform(self, df, columns=None):
        """
        Aplica fit e transform em sequência
        """
        self.fit(df, columns)
        return self.transform(df)
    
    def inverse_transform(self, df):
        """
        Reverte o escalonamento para os valores originais
        """
        df_inverse = df.copy()
        inverse_data = self.scaler.inverse_transform(df[self.column_names])
        df_inverse[self.column_names] = inverse_data
        return df_inverse
    
    def save(self, filepath):
        """
        Salva o scaler em disco
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
    
    @staticmethod
    def load(filepath):
        """
        Carrega o scaler do disco
        """
        with open(filepath, 'rb') as f:
            return pickle.load(f)


# Exemplo de uso
if __name__ == "__main__":
    # Dados de exemplo
    dados = pd.DataFrame({
        'id': [1, 2, 3, 4],
        'idade': [25, 30, 35, 40],
        'salario': [3000, 5000, 7000, 9000],
        'score': [0.1, 0.5, 0.8, 0.3]
    })
    
    print("Dados originais:")
    print(dados)
    print("\n" + "="*50)
    
    # Aplicar MinMaxScaler
    scaler = MinMaxScalerProcessor(feature_range=(0, 1))
    dados_scaled = scaler.fit_transform(dados, columns=['idade', 'salario', 'score'])
    
    print("Dados após MinMaxScaler:")
    print(dados_scaled)
    
    # Salvar scaler para uso posterior
    scaler.save('min_max_scaler.pkl')
    print("\nScaler salvo em 'min_max_scaler.pkl'")
    
    # Simular nova instância
    nova_instancia = pd.DataFrame({
        'id': [5],
        'idade': [28],
        'salario': [4500],
        'score': [0.6]
    })
    
    print("\n" + "="*50)
    print("Nova instância recebida:")
    print(nova_instancia)
    
    # Transformar nova instância
    nova_instancia_scaled = scaler.transform(nova_instancia)
    
    print("\nNova instância após escalonamento:")
    print(nova_instancia_scaled)
    
    # Exemplo de inversão
    dados_original = scaler.inverse_transform(dados_scaled)
    print("\nDados revertidos ao original (inverse_transform):")
    print(dados_original)