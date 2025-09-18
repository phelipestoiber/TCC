# src/ui/plotting.py

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Dict, Optional
import os

def plotar_curva_estabilidade(
    df_curva: pd.DataFrame,
    resultados_criterios: Dict,
    dados_condicao: Dict,
    nome_condicao: str,
    caminho_salvar: Optional[str] = None
):
    """
    Gera e exibe um gráfico da curva de estabilidade estática (GZ).

    Args:
        df_curva (pd.DataFrame): DataFrame com as colunas 'Angulo (°)', 'GZ (m)' e 'GZ Emborcador (m)'.
        resultados_criterios (Dict): Dicionário com os resultados da verificação de critérios.
        dados_condicao (Dict): Dicionário da condição de carga para obter o theta_f.
        nome_condicao (str): O nome da condição de carregamento para o título do gráfico.
        caminho_salvar (Optional[str]): Caminho opcional para salvar o gráfico como uma imagem.
    """
    try:
        # Extrai os ângulos e valores para plotagem
        angulos = df_curva['Angulo (°)']
        gz_righting = df_curva['GZ (m)']
        gz_heeling = df_curva['GZ Emborcador (m)']

        # Extrai os ângulos notáveis
        theta_1_str = resultados_criterios.get('Ângulo de Equilíbrio', {}).get('valor', '0.0°')
        theta_1 = float(theta_1_str.replace('°', ''))
        
        theta_f_str = resultados_criterios.get('Ângulo de Alagamento', {}).get('valor', '0.0°')
        theta_f = float(theta_f_str.replace('°', ''))

        # --- Início da Plotagem ---
        fig, ax = plt.subplots(figsize=(12, 7))

        # 1. Plotar curvas GZ e de Emborcamento
        ax.plot(angulos, gz_righting, label='Braço de Endireitamento (GZ)', color='blue', linewidth=2)
        ax.plot(angulos, gz_heeling, label='Braço Emborcador (Heeling Arm)', color='red', linestyle='--', linewidth=2)

        # 2. Adicionar linha de base
        ax.axhline(0, color='black', linewidth=0.5)

        # 3. Adicionar linhas verticais para ângulos notáveis
        ax.axvline(x=theta_1, color='green', linestyle=':', label=f'$\\theta_1$ (Equilíbrio) = {theta_1:.2f}°')
        ax.axvline(x=theta_f, color='purple', linestyle=':', label=f'$\\theta_f$ (Alagamento) = {theta_f:.2f}°')

        # 4. Sombrear as áreas A e B (Critério 2)
        # Área B (antes de theta_1)
        ax.fill_between(angulos, gz_heeling, gz_righting, where=(angulos <= theta_1), interpolate=True, color='red', alpha=0.2, label='Área B (Emborcamento)')
        # Área A (residual, entre theta_1 e min(40, theta_f))
        limite_area = min(40.0, theta_f)
        ax.fill_between(angulos, gz_heeling, gz_righting, where=(angulos >= theta_1) & (angulos <= limite_area), 
                        color='blue', alpha=0.2, label='Área A (Residual)')

        # 5. Estilização do Gráfico
        ax.set_title(f"Curva de Estabilidade Estática - {nome_condicao}", fontsize=16)
        ax.set_xlabel("Ângulo de Inclinação (°)", fontsize=12)
        ax.set_ylabel("Braço (m)", fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.set_xlim(left=0, right=max(60, theta_f + 5))
        ax.set_ylim(bottom=min(0, gz_righting.min()) - 0.1)

        # 6. Salvar e/ou mostrar o gráfico
        if caminho_salvar:
            try:
                os.makedirs(os.path.dirname(caminho_salvar), exist_ok=True)
                plt.savefig(caminho_salvar)
                print(f"-> Gráfico salvo em: '{caminho_salvar}'")
            except Exception as e:
                print(f"\nERRO ao salvar o gráfico: {e}")

        plt.show()

    except Exception as e:
        print(f"\nOcorreu um erro inesperado ao gerar o gráfico para '{nome_condicao}': {e}")
        import traceback
        traceback.print_exc()