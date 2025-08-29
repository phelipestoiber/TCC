import pandas as pd
from rich.console import Console
from rich.table import Table

def exibir_tabela_hidrostatica(df: pd.DataFrame):
    """
    Exibe um DataFrame de curvas hidrostáticas em uma tabela bem formatada no terminal.

    Args:
        df (pd.DataFrame): O DataFrame contendo os resultados dos cálculos.
    """
    console = Console()
    table = Table(title="--- Curvas Hidrostáticas ---", show_header=True, header_style="bold magenta")

    # Adiciona as colunas à tabela
    for column in df.columns:
        # Converte o nome da coluna para string antes de adicionar.
        # Isto é crucial para a tabela KN, onde os nomes das colunas são ângulos (floats).
        header = f"{column:.1f}" if isinstance(column, float) else str(column)
        
        # Justifica os números à direita e o resto à esquerda
        # justify = "center" if df[column].dtype in ['float64', 'int64'] else "left"
        justify = "center"
        table.add_column(header, justify=justify)

    # Adiciona as linhas com os dados formatados
    for _, row in df.iterrows():
        # Converte cada item da linha para uma string formatada
        # ._asdict() pode ser usado se iterrows() retornar tuplas nomeadas
        row_str = [f"{val:.4f}" if isinstance(val, float) else str(val) for val in row]
        table.add_row(*row_str)

    console.print(table)