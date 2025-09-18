# src/io/file_handler.py

import pandas as pd
import os
from typing import Any, Dict

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
    
    def salvar_relatorio_rpi(self, caminho_arquivo: str, calculadora_rpi: Any):
        """
        Salva um relatório de texto consolidado com os resultados finais
        da Prova de Inclinação.

        Args:
            caminho_arquivo (str): O caminho completo onde o arquivo .txt será salvo.
            calculadora_rpi (Any): A instância da classe CalculadoraRPI após
                                     todos os cálculos terem sido executados.
        """
        try:
            # Garante que a pasta de destino exista
            pasta_destino = os.path.dirname(caminho_arquivo)
            if not os.path.exists(pasta_destino):
                os.makedirs(pasta_destino)

            # Usamos 'with' para garantir que o arquivo seja fechado corretamente
            with open(caminho_arquivo, 'w', encoding='utf-8') as f:
                f.write("="*60 + "\n")
                f.write("RELATÓRIO FINAL DA PROVA DE INCLINAÇÃO\n")
                f.write("="*60 + "\n\n")

                # Bloco 1: Condição de Navio Leve
                if hasattr(calculadora_rpi, 'navio_leve_resultados'):
                    f.write("--- 1. CONDIÇÃO DE NAVIO LEVE ---\n")
                    for chave, valor in calculadora_rpi.navio_leve_resultados.items():
                        f.write(f"{chave:<28}: {valor:.4f}\n")
                    f.write("\n")
                else:
                    f.write("--- 1. CONDIÇÃO DE NAVIO LEVE (DADOS NÃO ENCONTRADOS) ---\n\n")

                # Bloco 2: Flutuação na Condição de Navio Leve
                if hasattr(calculadora_rpi, 'flutuacao_navio_leve'):
                    f.write("--- 2. FLUTUAÇÃO NA CONDIÇÃO DE NAVIO LEVE ---\n")
                    for chave, valor in calculadora_rpi.flutuacao_navio_leve.items():
                        f.write(f"{chave:<28}: {valor:.4f}\n")
                    f.write("\n")
                else:
                    f.write("--- 2. FLUTUAÇÃO (DADOS NÃO ENCONTRADOS) ---\n\n")

                # Bloco 3: Hidrostáticas na Condição de Navio Leve
                if hasattr(calculadora_rpi, 'hidrostaticos_navio_leve'):
                    f.write("--- 3. HIDROSTÁTICAS NA CONDIÇÃO DE NAVIO LEVE ---\n")
                    for chave, valor in calculadora_rpi.hidrostaticos_navio_leve.items():
                        f.write(f"{chave:<28}: {valor:.4f}\n")
                    f.write("\n")
                else:
                    f.write("--- 3. HIDROSTÁTICAS (DADOS NÃO ENCONTRADOS) ---\n\n")

            print(f"\n-> Relatório RPI salvo com sucesso em: '{caminho_arquivo}'")

        except Exception as e:
            print(f"\nERRO ao salvar o relatório RPI: {e}")

    def salvar_relatorio_eed(self, caminho_arquivo: str, calculadora_eed: Any, verificador: Any, resultados_verificacao: Dict):
        """
        Salva um relatório de texto completo com todos os resultados do
        Estudo de Estabilidade Definitivo (EED).

        Args:
            caminho_arquivo (str): O caminho completo onde o arquivo .txt será salvo.
            calculadora_eed (Any): A instância da classe CalculadoraEED após os cálculos.
            verificador (Any): A instância da classe VerificadorCriterios.
            resultados_verificacao (Dict): Dicionário com os resultados da verificação para cada condição.
        """
        try:
            pasta_destino = os.path.dirname(caminho_arquivo)
            os.makedirs(pasta_destino, exist_ok=True)

            with open(caminho_arquivo, 'w', encoding='utf-8') as f:
                f.write("="*80 + "\n")
                f.write("RELATÓRIO FINAL - ESTUDO DE ESTABILIDADE DEFINITIVO (EED)\n")
                f.write("="*80 + "\n")
                f.write(f"\nÁrea de Navegação Considerada: {verificador.area_navegação}\n")

                # Escreve a Tabela de Pesos e Centros completa (base para todos os cálculos)
                f.write("\n--- TABELA DE PESOS E CENTROS GERAL ---\n")
                df_pesos = pd.DataFrame(calculadora_eed.tabela_pesos)
                f.write(df_pesos.to_string(index=False))
                f.write("\n\n" + "="*80 + "\n")

                # Itera sobre cada condição de carregamento
                for nome_cond, dados_cond in calculadora_eed.resultados_condicoes.items():
                    f.write(f"\n--- ANÁLISE DA CONDIÇÃO: {nome_cond} ---\n")

                    # 1. Resumo da Condição
                    f.write("\n  1. Resumo da Condição de Carregamento:\n")
                    f.write(f"    - Peso Total (Deslocamento): {dados_cond.get('peso_total', 0.0):.4f} t\n")
                    f.write(f"    - LCG da Condição:             {dados_cond.get('lcg_condicao', 0.0):.4f} m\n")
                    f.write(f"    - KG da Condição:              {dados_cond.get('kg_condicao', 0.0):.4f} m\n")

                    # 2. Características Hidrostáticas
                    if 'hidrostaticos' in dados_cond:
                        f.write("\n  2. Características Hidrostáticas:\n")
                        for chave, valor in dados_cond['hidrostaticos'].items():
                            f.write(f"    - {chave:<28}: {valor:.4f}\n")

                    # 3. Curva de Estabilidade (GZ)
                    if 'curva_gz' in dados_cond:
                        f.write("\n  3. Curva de Estabilidade Estática (GZ):\n")
                        # Para economizar espaço, podemos mostrar apenas alguns pontos ou a tabela completa
                        f.write(dados_cond['curva_gz'].to_string(index=False))
                        f.write("\n")

                    # 4. Verificação dos Critérios
                    if nome_cond in resultados_verificacao:
                        f.write("\n  4. Verificação de Critérios de Estabilidade:\n")
                        for criterio, resultado in resultados_verificacao[nome_cond].items():
                            status = "PASSOU" if resultado['passou'] else "FALHOU"
                            linha = f"    - {criterio:<22} | Valor: {resultado['valor']:<18} | Esperado: {resultado['esperado']:<18} | Status: {status}\n"
                            f.write(linha)
                    
                    f.write("\n" + "="*80 + "\n")

            print(f"\n-> Relatório EED salvo com sucesso em: '{caminho_arquivo}'")

        except Exception as e:
            print(f"\nERRO ao salvar o relatório EED: {e}")