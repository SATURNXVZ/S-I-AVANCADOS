import pandas as pd
import numpy as np
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Importando os módulos criados
from NORMALIZAÇÃO.onehot import OneHotEncoderProcessor
from NORMALIZAÇÃO.minmaxscaler import MinMaxScalerProcessor
from NORMALIZAÇÃO.label import LabelEncoderProcessor

class ProcessadorDadosCompleto:
    def __init__(self):
        self.one_hot_encoder = OneHotEncoderProcessor()
        self.label_encoder = LabelEncoderProcessor()
        self.scaler = MinMaxScalerProcessor()
        self.processed_columns = {}
        self.original_columns = None
        self.numeric_columns = []
        self.categorical_columns = []
        self.one_hot_columns = []
        self.label_columns = []
        
    def identificar_tipos_colunas(self, df):
        """
        Identifica automaticamente os tipos de colunas
        """
        self.original_columns = df.columns.tolist()
        
        # Identificar colunas numéricas
        self.numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Identificar colunas categóricas (object, category)
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Separar colunas para One-Hot (nominais) e Label (ordinais)
        # Por padrão, colunas com menos de 10 categorias únicas podem ser one-hot
        # Colunas com mais categorias podem ser label encoding
        self.one_hot_columns = []
        self.label_columns = []
        
        for col in categorical_cols:
            unique_count = df[col].nunique()
            if unique_count <= 10:  # Se poucas categorias, usa one-hot
                self.one_hot_columns.append(col)
            else:  # Se muitas categorias, usa label encoding
                self.label_columns.append(col)
        
        return {
            'numeric': self.numeric_columns,
            'one_hot': self.one_hot_columns,
            'label': self.label_columns
        }
    
    def aplicar_one_hot_encoding(self, df):
        """
        Aplica One-Hot Encoding nas colunas identificadas
        """
        if not self.one_hot_columns:
            return df
        
        df_resultado = df.copy()
        
        for col in self.one_hot_columns:
            if col in df_resultado.columns:
                print(f"\nAplicando One-Hot Encoding na coluna: {col}")
                encoded_data = self.one_hot_encoder.fit_transform(df_resultado, col)
                
                # Remover coluna original e adicionar as codificadas
                df_resultado = df_resultado.drop(col, axis=1)
                df_resultado = pd.concat([df_resultado, encoded_data], axis=1)
                
                # Armazenar informação
                self.processed_columns[col] = {
                    'type': 'one_hot',
                    'new_columns': encoded_data.columns.tolist()
                }
        
        return df_resultado
    
    def aplicar_label_encoding(self, df):
        """
        Aplica Label Encoding nas colunas identificadas
        """
        if not self.label_columns:
            return df
        
        print(f"\nAplicando Label Encoding nas colunas: {self.label_columns}")
        df_resultado = self.label_encoder.fit_transform(df, self.label_columns)
        
        for col in self.label_columns:
            self.processed_columns[col] = {
                'type': 'label',
                'new_column': col
            }
        
        return df_resultado
    
    def aplicar_min_max_scaler(self, df):
        """
        Aplica MinMaxScaler nas colunas numéricas
        """
        if not self.numeric_columns:
            return df
        
        print(f"\nAplicando MinMaxScaler nas colunas numéricas: {self.numeric_columns}")
        df_resultado = self.scaler.fit_transform(df)
        
        for col in self.numeric_columns:
            self.processed_columns[col] = {
                'type': 'scaled',
                'original_column': col
            }
        
        return df_resultado
    
    def processar_completo(self, df, aplicar_one_hot=True, aplicar_label=True, aplicar_scaler=True):
        """
        Processa o DataFrame completo com todas as transformações
        
        Args:
            df: DataFrame original
            aplicar_one_hot: Aplicar One-Hot Encoding
            aplicar_label: Aplicar Label Encoding
            aplicar_scaler: Aplicar MinMaxScaler
        
        Returns:
            DataFrame processado
        """
        print("="*80)
        print("INICIANDO PROCESSAMENTO DE DADOS")
        print("="*80)
        
        # Identificar tipos de colunas
        tipos = self.identificar_tipos_colunas(df)
        
        print("\n📊 ESTRUTURA ORIGINAL DOS DADOS:")
        print(f"Total de colunas: {len(df.columns)}")
        print(f"Total de linhas: {len(df)}")
        print(f"\nColunas numéricas ({len(tipos['numeric'])}): {tipos['numeric']}")
        print(f"Colunas para One-Hot Encoding ({len(tipos['one_hot'])}): {tipos['one_hot']}")
        print(f"Colunas para Label Encoding ({len(tipos['label'])}): {tipos['label']}")
        
        # Mostrar informações das colunas categóricas
        if tipos['one_hot']:
            print("\nDetalhes das colunas para One-Hot Encoding:")
            for col in tipos['one_hot']:
                print(f"  • {col}: {df[col].nunique()} categorias únicas")
                print(f"    Exemplos: {df[col].unique()[:5]}")
        
        if tipos['label']:
            print("\nDetalhes das colunas para Label Encoding:")
            for col in tipos['label']:
                print(f"  • {col}: {df[col].nunique()} categorias únicas")
                print(f"    Exemplos: {df[col].unique()[:5]}")
        
        if tipos['numeric']:
            print("\nEstatísticas das colunas numéricas:")
            for col in tipos['numeric']:
                print(f"  • {col}: min={df[col].min():.2f}, max={df[col].max():.2f}, mean={df[col].mean():.2f}")
        
        df_processado = df.copy()
        
        # Aplicar transformações na ordem correta
        if aplicar_one_hot:
            df_processado = self.aplicar_one_hot_encoding(df_processado)
        
        if aplicar_label:
            df_processado = self.aplicar_label_encoding(df_processado)
        
        if aplicar_scaler:
            df_processado = self.aplicar_min_max_scaler(df_processado)
        
        print("="*80)
        print(f"\n ESTRUTURA FINAL DOS DADOS:")
        print(f"Total de colunas: {len(df_processado.columns)}")
        print(f"Total de linhas: {len(df_processado)}")
        print(f"\nColunas finais: {df_processado.columns.tolist()}")
        
        return df_processado
    
    def salvar_processadores(self, caminho_base):
        """
        Salva todos os processadores treinados
        """
        try:
            self.one_hot_encoder.save(f"{caminho_base}/one_hot_encoder.pkl")
            self.label_encoder.save(f"{caminho_base}/label_encoder.pkl")
            self.scaler.save(f"{caminho_base}/min_max_scaler.pkl")
            print(f"\nProcessadores salvos em: {caminho_base}")
        except Exception as e:
            print(f"Erro ao salvar processadores: {e}")
    
    def carregar_processadores(self, caminho_base):
        """
        Carrega processadores salvos anteriormente
        """
        try:
            self.one_hot_encoder = OneHotEncoderProcessor.load(f"{caminho_base}/one_hot_encoder.pkl")
            self.label_encoder = LabelEncoderProcessor.load(f"{caminho_base}/label_encoder.pkl")
            self.scaler = MinMaxScalerProcessor.load(f"{caminho_base}/min_max_scaler.pkl")
            print(f"\nProcessadores carregados de: {caminho_base}")
            return True
        except Exception as e:
            print(f"Erro ao carregar processadores: {e}")
            return False


def ler_csv_do_download(nome_arquivo):
    """
    Lê um arquivo CSV do diretório Downloads
    
    Args:
        nome_arquivo: Nome do arquivo CSV (ex: 'dados.csv')
    
    Returns:
        DataFrame com os dados
    """
    # Caminho base do Downloads
    caminho_downloads = r"C:\Users\VITORHENRIQUEDEMELO\Downloads"
    
    # Construir caminho completo
    caminho_completo = os.path.join(caminho_downloads, nome_arquivo)
    
    '''# Verificar se arquivo existe
    if not os.path.exists(caminho_completo):
        print(f"❌ Arquivo não encontrado: {caminho_completo}")
        
        # Listar arquivos CSV no diretório
        arquivos_csv = [f for f in os.listdir(caminho_downloads) if f.endswith('.csv')]
        if arquivos_csv:
            print(f"\nArquivos CSV encontrados no diretório:")
            for arquivo in arquivos_csv:
                print(f"  • {arquivo}")
        return None
    '''
    try:
        print(f"Lendo arquivo: {caminho_completo}")
        df = pd.read_csv(caminho_completo, encoding='utf-8')
        print(f"Arquivo carregado com sucesso!")
        print(f"Dimensões: {df.shape[0]} linhas x {df.shape[1]} colunas")
        return df
    except UnicodeDecodeError:
        # Tentar com outra codificação
        print("Tentando com encoding 'latin1'...")
        df = pd.read_csv(caminho_completo, encoding='latin1')
        print(f"Arquivo carregado com sucesso!")
        return df
    except Exception as e:
        print(f"Erro ao ler arquivo: {e}")
        return None


def main():
    """
    Função principal
    """
    print("="*80)
    print("SISTEMA DE PRÉ-PROCESSAMENTO DE DADOS")
    print("="*80)
    
    nome_arquivo = "dados_normalizar.csv"
    # Ler arquivo CSV
    df_original = ler_csv_do_download(nome_arquivo)
    
    if df_original is None:
        return
    
    # Mostrar amostra dos dados
    print("\n" + "="*80)
    print("AMOSTRA DOS DADOS ORIGINAIS:")
    print("="*80)
    print(df_original.head(10))
    
    print("\nINFORMAÇÕES DO DATAFRAME:")
    print("="*80)
    print(df_original.info())
    
    # Perguntar se deseja processar
    resposta = input("\nDeseja processar os dados? (S/N): ").strip().upper()
    
    if resposta != 'S':
        print("Processamento cancelado!")
        return
    
    # Criar processador
    processador = ProcessadorDadosCompleto()
    
    # Processar dados
    df_processado = processador.processar_completo(df_original)
    
    # Mostrar resultados
    print("\n" + "="*80)
    print("AMOSTRA DOS DADOS PROCESSADOS:")
    print("="*80)
    print(df_processado.head(10))
    
    print("\nESTATÍSTICAS DOS DADOS PROCESSADOS:")
    print("="*80)
    print(df_processado.describe())
    
    # Salvar dados processados
    caminho_saida = r"C:\Users\VITORHENRIQUEDEMELO\Downloads"
    nome_saida = nome_arquivo.replace('.csv', '_processado.csv')
    caminho_completo_saida = os.path.join(caminho_saida, nome_saida)
    
    df_processado.to_csv(caminho_completo_saida, index=False)
    print(f"\nDados processados salvos em: {caminho_completo_saida}")
    
    # Salvar processadores treinados
    caminho_processadores = r"C:\Users\VITORHENRIQUEDEMELO\Downloads\processadores"
    os.makedirs(caminho_processadores, exist_ok=True)
    processador.salvar_processadores(caminho_processadores)
    
    # Perguntar se deseja processar outra instância
    resposta2 = input("\nDeseja testar o processamento de uma nova instância? (S/N): ").strip().upper()
    
    if resposta2 == 'S':
        print("\n" + "="*80)
        print("TESTE COM NOVA INSTÂNCIA")
        print("="*80)
        
        # Criar exemplo de nova instância
        print("\nCriando uma nova instância de exemplo...")
        
        # Criar dicionário com valores para cada coluna original
        nova_instancia = {}
        for col in df_original.columns:
            if col in processador.numeric_columns:
                valor = input(f"Digite o valor para '{col}' (numérico): ")
                nova_instancia[col] = [float(valor) if valor else 0]
            else:
                print(f"Valores possíveis para '{col}': {df_original[col].unique()[:5]}")
                valor = input(f"Digite o valor para '{col}': ")
                nova_instancia[col] = [valor if valor else 'desconhecido']
        
        df_nova = pd.DataFrame(nova_instancia)
        print("\nNova instância criada:")
        print(df_nova)
        
        # Aplicar transformações na nova instância
        df_nova_processada = df_nova.copy()
        
        # Aplicar one-hot encoding (usando os encoders já treinados)
        for col in processador.one_hot_columns:
            if col in df_nova_processada.columns:
                encoded_data = processador.one_hot_encoder.transform(df_nova_processada, col)
                df_nova_processada = df_nova_processada.drop(col, axis=1)
                df_nova_processada = pd.concat([df_nova_processada, encoded_data], axis=1)
        
        # Aplicar label encoding
        if processador.label_columns:
            df_nova_processada = processador.label_encoder.transform(df_nova_processada)
        
        # Aplicar scaler
        if processador.numeric_columns:
            df_nova_processada = processador.scaler.transform(df_nova_processada)
        
        print("\nNova instância após processamento:")
        print(df_nova_processada)
    
    print("\n" + "="*80)
    print("✨ PROCESSAMENTO FINALIZADO COM SUCESSO!")
    print("="*80)


if __name__ == "__main__":
    main()