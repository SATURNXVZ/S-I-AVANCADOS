import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pickle

class LabelEncoderProcessor:
    def __init__(self):
        self.encoders = {}
        self.original_dtypes = {}  #armazenar os tipos originais das colunas
    
    
    #aplica fit para colunas especificas
    def fit(self, df, columns):
        for column in columns:
            if column in df.columns:
                # Armazenar o tipo original da coluna
                self.original_dtypes[column] = df[column].dtype
                
                encoder = LabelEncoder()
                encoder.fit(df[column].astype(str))
                self.encoders[column] = encoder
        
        return self
    
    #aplica transform
    def transform(self, df):
        df_encoded = df.copy()
        
        for column, encoder in self.encoders.items():
            if column in df_encoded.columns:
                # Trata valores não vistos anteriormente
                df_encoded[column] = df_encoded[column].astype(str)
                unknown_mask = ~df_encoded[column].isin(encoder.classes_)
                
                # Aplica encoding
                encoded_values = encoder.transform(df_encoded[column])
                df_encoded[column] = encoded_values
                
                # Atribuir -1 para valores desconhecidos
                if unknown_mask.any():
                    df_encoded.loc[unknown_mask, column] = -1
        
        return df_encoded
    
    
    def fit_transform(self, df, columns):
        self.fit(df, columns)
        return self.transform(df)
    
    
    #reverete o encoding para valores originais
    def inverse_transform(self, df):
        df_inverse = df.copy()
        
        for column, encoder in self.encoders.items():
            if column in df_inverse.columns:
                #criar uma nova coluna para os valores decodificados
                decoded_values = pd.Series(index=df_inverse.index, dtype='object')
                
                #filtra valores que não são -1 (valores desconhecidos)
                mask = df_inverse[column] != -1
                if mask.any():
                    decoded = encoder.inverse_transform(df_inverse.loc[mask, column].astype(int))
                    decoded_values.loc[mask] = decoded
                
                #manter valores desconhecidos (-1) como ja estao
                if (~mask).any():
                    decoded_values.loc[~mask] = df_inverse.loc[~mask, column]
                
                #converte para o tipo original se necessario
                if column in self.original_dtypes:
                    if pd.api.types.is_categorical_dtype(self.original_dtypes[column]):
                        decoded_values = decoded_values.astype(self.original_dtypes[column])
                    elif pd.api.types.is_object_dtype(self.original_dtypes[column]):
                        decoded_values = decoded_values.astype(object)
                
                df_inverse[column] = decoded_values
        
        return df_inverse
    
    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
    
    @staticmethod
    def load(filepath):
        with open(filepath, 'rb') as f:
            return pickle.load(f)
