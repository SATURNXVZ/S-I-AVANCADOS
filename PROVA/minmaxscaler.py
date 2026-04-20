import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pickle


#inicializa minmax scaler, e define o intervalo (padrão)
class MinMaxScalerProcessor:
    def __init__(self, feature_range=(0, 1)):
        self.scaler = MinMaxScaler(feature_range=feature_range)
        self.column_names = None
    
    #aplica fit, recebe columns q é a lista pra escalonar, se none = todas numericas
    def fit(self, df, columns=None):
        if columns is None:
            #seleciona apenas colunas numéricas
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        self.column_names = columns
        self.scaler.fit(df[columns])
        return self
    
    #transforma os dados, e devolve o dataframe com dados escalonados
    def transform(self, df):
        df_scaled = df.copy()
        scaled_data = self.scaler.transform(df[self.column_names])
        df_scaled[self.column_names] = scaled_data
        return df_scaled
    
    #aplica fit e transform
    def fit_transform(self, df, columns=None):
        self.fit(df, columns)
        return self.transform(df)
    
    #devolve os valores originais
    def inverse_transform(self, df):
        df_inverse = df.copy()
        inverse_data = self.scaler.inverse_transform(df[self.column_names])
        df_inverse[self.column_names] = inverse_data
        return df_inverse
    
    
    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
    
    @staticmethod
    def load(filepath):
        with open(filepath, 'rb') as f:
            return pickle.load(f)
