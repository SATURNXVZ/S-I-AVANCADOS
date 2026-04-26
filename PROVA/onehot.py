import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import pickle

class OneHotEncoderProcessor:
    def __init__(self):
        self.encoder = None
        self.feature_names = None
        self.column_name = None  #guarda qual coluna foi usada
    
    
    #aplica fit em uma coluna especifica categórica
    def fit(self, df, column_name):
        self.column_name = column_name
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        
        data = df[[column_name]]
        self.encoder.fit(data)
        
        #gera nome das colunas após onehot
        self.feature_names = [f"{column_name}_{category}" for category in self.encoder.categories_[0]]
        
        return self
    
    #transforma os dados aplicando one-hot encoding, recebe o dataframe e devolve as nova colunas codificadas
    def transform(self, df):
        if self.encoder is None or self.column_name is None:
            raise ValueError("Encoder não foi ajustado. Chame fit primeiro.")
        
        data = df[[self.column_name]]
        encoded_data = self.encoder.transform(data)
        
        #criar DataFrame com as colunas codificadas
        encoded_df = pd.DataFrame(
            encoded_data,
            columns=self.feature_names,
            index=df.index
        )
        
        return encoded_df
    
    #aplica fit e transform
    def fit_transform(self, df, column_name):
        self.fit(df, column_name)
        return self.transform(df)
     
    #salva o encoder (pkl)
    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
    
    #carrega o encoder
    @staticmethod
    def load(filepath):
        with open(filepath, 'rb') as f:
            return pickle.load(f)


