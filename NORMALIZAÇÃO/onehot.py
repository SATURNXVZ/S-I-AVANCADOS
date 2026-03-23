import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import pickle

class OneHotEncoderProcessor:
    def __init__(self):
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self.feature_names = None
    
    def fit(self, df, column_name):
        """
        Aplica o fit no OneHotEncoder para uma coluna específica
        
        Args:
            df: DataFrame com os dados
            column_name: Nome da coluna categórica
        """
        data = df[[column_name]]
        self.encoder.fit(data)
        
        # Gerar nomes das colunas após one-hot encoding
        self.feature_names = [f"{column_name}_{category}" for category in self.encoder.categories_[0]]
        
        return self
    
    def transform(self, df, column_name):
        """
        Transforma os dados aplicando one-hot encoding
        
        Args:
            df: DataFrame com os dados
            column_name: Nome da coluna categórica
        
        Returns:
            DataFrame com as novas colunas codificadas
        """
        data = df[[column_name]]
        encoded_data = self.encoder.transform(data)
        
        # Criar DataFrame com as colunas codificadas
        encoded_df = pd.DataFrame(
            encoded_data,
            columns=self.feature_names,
            index=df.index
        )
        
        return encoded_df
    
    def fit_transform(self, df, column_name):
        """
        Aplica fit e transform em sequência
        """
        self.fit(df, column_name)
        return self.transform(df, column_name)
    
    def save(self, filepath):
        """
        Salva o encoder em disco
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
    
    @staticmethod
    def load(filepath):
        """
        Carrega o encoder do disco
        """
        with open(filepath, 'rb') as f:
            return pickle.load(f)


# Exemplo de uso
if __name__ == "__main__":
    # Dados de exemplo
    dados = pd.DataFrame({
        'id': [1, 2, 3, 4],
        'cor': ['Azul', 'Verde', 'Vermelho', 'Azul'],
        'preco': [100, 150, 200, 120]
    })
    
    print("Dados originais:")
    print(dados)
    print("\n" + "="*50)
    
    # Aplicar one-hot encoding
    encoder = OneHotEncoderProcessor()
    encoded_colors = encoder.fit_transform(dados, 'cor')
    
    # Remover coluna original e concatenar
    dados_final = dados.drop('cor', axis=1)
    dados_final = pd.concat([dados_final, encoded_colors], axis=1)
    
    print("Dados após one-hot encoding:")
    print(dados_final)
    
    # Salvar encoder para uso posterior
    encoder.save('one_hot_encoder.pkl')
    print("\nEncoder salvo em 'one_hot_encoder.pkl'")
    
    # Simular nova instância
    nova_instancia = pd.DataFrame({
        'id': [5],
        'cor': ['Azul'],
        'preco': [180]
    })
    
    print("\n" + "="*50)
    print("Nova instância recebida:")
    print(nova_instancia)
    
    # Transformar nova instância
    encoded_new = encoder.transform(nova_instancia, 'cor')
    nova_instancia_final = nova_instancia.drop('cor', axis=1)
    nova_instancia_final = pd.concat([nova_instancia_final, encoded_new], axis=1)
    
    print("\nNova instância após transformação:")
    print(nova_instancia_final)