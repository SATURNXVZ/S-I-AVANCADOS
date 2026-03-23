import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pickle

class LabelEncoderProcessor:
    def __init__(self):
        self.encoders = {}
        self.encoded_columns = {}
        self.original_dtypes = {}  # Armazenar os tipos originais das colunas
    
    def fit(self, df, columns):
        """
        Aplica fit nos LabelEncoders para colunas específicas
        
        Args:
            df: DataFrame com os dados
            columns: Lista de colunas categóricas para codificar
        """
        for column in columns:
            if column in df.columns:
                # Armazenar o tipo original da coluna
                self.original_dtypes[column] = df[column].dtype
                
                encoder = LabelEncoder()
                encoder.fit(df[column].astype(str))
                self.encoders[column] = encoder
                self.encoded_columns[column] = list(encoder.classes_)
        
        return self
    
    def transform(self, df):
        """
        Transforma os dados aplicando Label Encoding
        
        Args:
            df: DataFrame com os dados
        
        Returns:
            DataFrame com as colunas codificadas
        """
        df_encoded = df.copy()
        
        for column, encoder in self.encoders.items():
            if column in df_encoded.columns:
                # Trata valores não vistos anteriormente
                df_encoded[column] = df_encoded[column].astype(str)
                unknown_mask = ~df_encoded[column].isin(encoder.classes_)
                
                # Aplica encoding
                encoded_values = encoder.transform(df_encoded[column])
                df_encoded[column] = encoded_values
                
                # Opcional: atribuir -1 para valores desconhecidos
                if unknown_mask.any():
                    df_encoded.loc[unknown_mask, column] = -1
        
        return df_encoded
    
    def fit_transform(self, df, columns):
        """
        Aplica fit e transform em sequência
        """
        self.fit(df, columns)
        return self.transform(df)
    
    def inverse_transform(self, df):
        """
        Reverte o encoding para os valores originais
        """
        df_inverse = df.copy()
        
        for column, encoder in self.encoders.items():
            if column in df_inverse.columns:
                # Criar uma nova coluna temporária para os valores decodificados
                decoded_values = pd.Series(index=df_inverse.index, dtype='object')
                
                # Filtra valores que não são -1 (valores desconhecidos)
                mask = df_inverse[column] != -1
                if mask.any():
                    # Decodificar os valores
                    decoded = encoder.inverse_transform(df_inverse.loc[mask, column].astype(int))
                    decoded_values.loc[mask] = decoded
                
                # Para valores desconhecidos (-1), manter como estão
                if (~mask).any():
                    decoded_values.loc[~mask] = df_inverse.loc[~mask, column]
                
                # Converter para o tipo original se necessário
                if column in self.original_dtypes:
                    if pd.api.types.is_categorical_dtype(self.original_dtypes[column]):
                        decoded_values = decoded_values.astype(self.original_dtypes[column])
                    elif pd.api.types.is_object_dtype(self.original_dtypes[column]):
                        decoded_values = decoded_values.astype(object)
                
                # Atribuir a coluna decodificada
                df_inverse[column] = decoded_values
        
        return df_inverse
    
    def save(self, filepath):
        """
        Salva os encoders em disco
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
    
    @staticmethod
    def load(filepath):
        """
        Carrega os encoders do disco
        """
        with open(filepath, 'rb') as f:
            return pickle.load(f)


# Exemplo de uso
if __name__ == "__main__":
    # Dados de exemplo
    dados = pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'categoria': ['A', 'B', 'A', 'C', 'B'],
        'tamanho': ['P', 'M', 'G', 'M', 'P'],
        'status': ['ativo', 'inativo', 'ativo', 'ativo', 'inativo']
    })
    
    print("Dados originais:")
    print(dados)
    print("\n" + "="*50)
    print("\nTipos de dados originais:")
    print(dados.dtypes)
    print("\n" + "="*50)
    
    # Aplicar Label Encoding
    label_encoder = LabelEncoderProcessor()
    dados_encoded = label_encoder.fit_transform(dados, ['categoria', 'tamanho', 'status'])
    
    print("\nDados após Label Encoding:")
    print(dados_encoded)
    print("\nTipos após encoding:")
    print(dados_encoded.dtypes)
    
    # Salvar encoders para uso posterior
    label_encoder.save('label_encoder.pkl')
    print("\nEncoders salvos em 'label_encoder.pkl'")
    
    # Simular nova instância
    nova_instancia = pd.DataFrame({
        'id': [6],
        'categoria': ['B'],
        'tamanho': ['G'],
        'status': ['ativo']
    })
    
    print("\n" + "="*50)
    print("Nova instância recebida:")
    print(nova_instancia)
    
    # Transformar nova instância
    nova_instancia_encoded = label_encoder.transform(nova_instancia)
    
    print("\nNova instância após encoding:")
    print(nova_instancia_encoded)
    
    # Exemplo de inversão - sem warnings agora
    dados_original = label_encoder.inverse_transform(dados_encoded)
    print("\n" + "="*50)
    print("Dados revertidos ao original (inverse_transform):")
    print(dados_original)
    print("\nTipos após inversão:")
    print(dados_original.dtypes)
    
    # Teste adicional: mostrar que os dados originais foram recuperados corretamente
    print("\n" + "="*50)
    print("Verificação de igualdade dos dados originais:")
    print("Original vs Revertido (categoria):", (dados['categoria'] == dados_original['categoria']).all())
    print("Original vs Revertido (tamanho):", (dados['tamanho'] == dados_original['tamanho']).all())
    print("Original vs Revertido (status):", (dados['status'] == dados_original['status']).all())