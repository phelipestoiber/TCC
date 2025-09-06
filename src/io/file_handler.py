# src/io/file_handler.py

import pandas as pd
import os

class FileHandler:
    """
    Responsável pela leitura e processamento de arquivos de dados.
    """

    def ler_tabela_cotas(self, caminho_arquivo: str) -> pd.DataFrame:
        """
        Lê um arquivo CSV da tabela de cotas e o carrega em um DataFrame.

        Args:
            caminho_arquivo (str): O caminho para o arquivo .csv.

        Returns:
            pd.DataFrame: DataFrame com os dados da tabela de cotas.
        
        Raises:
            FileNotFoundError: Se o arquivo não for encontrado.
            ValueError: Se o CSV não contiver as colunas esperadas ('x', 'y', 'z').
        """
        try:
            df = pd.read_csv(caminho_arquivo)
            # Normaliza os nomes das colunas para minúsculas e remove espaços
            df.columns = df.columns.str.strip().str.lower()
            
            colunas_esperadas = {'x', 'y', 'z'}
            if not colunas_esperadas.issubset(df.columns):
                raise ValueError(f"O arquivo CSV deve conter as colunas: {colunas_esperadas}")
            
            return df[list(colunas_esperadas)] # Garante a ordem

        except FileNotFoundError:
            print(f"Erro: Arquivo não encontrado em '{caminho_arquivo}'")
            raise
        except Exception as e:
            print(f"Ocorreu um erro ao ler o arquivo CSV: {e}")
            raise

    def processar_dados_balizas(self, 
                                tabela_cotas: pd.DataFrame, 
                                lpp: float, 
                                referencial_saida: str) -> pd.DataFrame:
        """
        Processa, valida e limpa a tabela de cotas da embarcação.
        Assume que o CSV de entrada tem a Perpendicular de Ré (AP) como origem para 'x'.

        Args:
            tabela_cotas (pd.DataFrame): DataFrame com os dados brutos.
            lpp (float): Comprimento entre perpendiculares (LPP) [m].
            referencial_saida (str): O referencial desejado para os resultados ('Perpendicular' ou 'Meio-navio').

        Returns:
            pd.DataFrame: A tabela de cotas processada e pronta para os cálculos.
            
        Raises:
            ValueError: Se uma baliza (estação 'x') intermediária possuir apenas um ponto.
        """
        df = tabela_cotas.copy()
        
        # 1. Ordenação
        df = df.sort_values(by=['x']).reset_index(drop=True)

        # 2. Transformação do Referencial de Saída
        # Assumimos que a entrada é sempre com x=0 na AP.
        # Se o usuário quer a saída referenciada ao meio-navio (MS), deslocamos os 'x'.
        if referencial_saida == 'Meio-navio':
            df['x'] -= lpp / 2
        
        # 3. Limpeza de Dados Duplicados
        # Arredondar para evitar problemas de ponto flutuante
        df = df.round({'x': 4, 'y': 4, 'z': 4})
        df = df.drop_duplicates(subset=['x', 'y', 'z'], keep='first')

        # 4. Validação de Estações com Ponto Único
        contagem_pontos = df['x'].value_counts()
        
        # Se houver mais de uma estação, identificamos as extremas
        if len(contagem_pontos) > 1:
            x_min = contagem_pontos.index.min()
            x_max = contagem_pontos.index.max()
            
            # Filtra estações que não são as extremas e que têm menos de 2 pontos
            estacoes_problematicas = contagem_pontos[
                (contagem_pontos < 2) & 
                (contagem_pontos.index != x_min) & 
                (contagem_pontos.index != x_max)
            ]
            
            if not estacoes_problematicas.empty:
                lista_estacoes = ", ".join(map(str, estacoes_problematicas.index.tolist()))
                raise ValueError(f"Erro Crítico: As estações intermediárias {lista_estacoes} possuem apenas 1 ponto.")

        print("Validação da tabela de cotas concluída com sucesso.")
        return df
    
    def salvar_resultados_csv(self, df_resultados: pd.DataFrame, caminho_arquivo: str):
        """
        Salva o DataFrame de resultados em um arquivo CSV.

        Args:
            df_resultados (pd.DataFrame): O DataFrame com os dados a serem salvos.
            caminho_arquivo (str): O caminho completo do arquivo onde os dados serão salvos.
        
        Raises:
            IOError: Se ocorrer um problema ao tentar escrever o arquivo (ex: permissão negada).
        """
        try:
            # Garante que o diretório de destino exista
            diretorio_destino = os.path.dirname(caminho_arquivo)
            if not os.path.exists(diretorio_destino):
                os.makedirs(diretorio_destino)

            # Salva o DataFrame em CSV
            # index=False: para não salvar o índice do DataFrame como uma coluna.
            # float_format: para garantir uma formatação consistente dos números.
            df_resultados.to_csv(caminho_arquivo, index=False, float_format='%.4f')
            print(f"\n-> Resultados salvos com sucesso em: '{caminho_arquivo}'")

        except IOError as e:
            print(f"\nErro ao salvar o arquivo: {e}")
            print("Verifique as permissões de escrita no diretório.")
            raise