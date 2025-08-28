import numpy as np
from typing import List, Dict, Any

def gerar_lista_de_calados(dados_calado: Dict[str, Any]) -> List[float]:
    """
    Gera uma lista de calados com base no método e nos valores
    fornecidos pelo usuário.

    Args:
        dados_calado (Dict[str, Any]): O dicionário de calados vindo do menu.

    Returns:
        List[float]: Uma lista ordenada de calados para os cálculos.
    """
    metodo = dados_calado.get("metodo")
    
    if metodo == "lista":
        # Converte a string "0.5; 1.0; 1.5" em uma lista de floats
        valores_str = dados_calado.get("valores", "")
        calados = [float(c.strip()) for c in valores_str.split(';') if c.strip()]
    
    elif metodo == "numero":
        # Gera N calados igualmente espaçados
        calados = np.linspace(
            dados_calado["min"], 
            dados_calado["max"], 
            int(dados_calado["num"])
        ).tolist()

    elif metodo == "passo":
        # Gera calados com um passo definido
        calados = np.arange(
            dados_calado["min"], 
            dados_calado["max"] + dados_calado["passo"]/2, # Garante inclusão do limite superior
            dados_calado["passo"]
        ).tolist()
        # Garante que o calado máximo seja o último, se não for incluído pelo passo
        if abs(calados[-1] - dados_calado["max"]) > 1e-6:
             calados.append(dados_calado["max"])
    
    else:
        calados = []

    # Retorna a lista ordenada e sem duplicatas
    return sorted(list(set(c for c in calados if c > 0)))