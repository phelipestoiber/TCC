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
    return sorted(list(set(c for c in calados if c >= 0)))

def gerar_lista_deslocamentos(dados: Dict[str, Any]) -> List[float]:
    """
    Gera uma lista de deslocamentos com base no método e nos valores
    fornecidos pelo utilizador.

    Args:
        dados (Dict[str, Any]): O dicionário de deslocamentos vindo do menu.

    Returns:
        List[float]: Uma lista ordenada de deslocamentos para os cálculos.
    """
    metodo = dados.get("metodo")
    deslocamentos = []
    
    if metodo == "lista":
        # Converte a string "100; 200; 300" numa lista de floats.
        valores_str = dados.get("valores", "")
        deslocamentos = [float(c.strip()) for c in valores_str.split(';') if c.strip()]
    
    elif metodo == "numero":
        # Gera N deslocamentos igualmente espaçados entre o mínimo e o máximo.
        deslocamentos = np.linspace(
            dados["min"], 
            dados["max"], 
            int(dados["num"])
        ).tolist()

    elif metodo == "passo":
        # Gera deslocamentos com um passo definido.
        deslocamentos = np.arange(
            dados["min"], 
            dados["max"] + dados["passo"]/2, # Garante a inclusão do limite superior.
            dados["passo"]
        ).tolist()
    
    # Retorna a lista ordenada, sem duplicatas e com valores não negativos.
    return sorted(list(set(d for d in deslocamentos if d >= 0)))

def gerar_lista_angulos(dados: Dict[str, Any]) -> List[float]:
    """
    Gera uma lista de ângulos com base no método e nos valores
    fornecidos pelo utilizador.

    Args:
        dados (Dict[str, Any]): O dicionário de ângulos vindo do menu.

    Returns:
        List[float]: Uma lista ordenada de ângulos para os cálculos.
    """
    metodo = dados.get("metodo")
    angulos = []
    
    if metodo == "lista":
        # Converte a string "0; 10; 20" numa lista de floats.
        valores_str = dados.get("valores", "")
        angulos = [float(c.strip()) for c in valores_str.split(';') if c.strip()]
    
    elif metodo == "numero":
        # Gera N ângulos igualmente espaçados entre o mínimo e o máximo.
        angulos = np.linspace(
            dados["min"], 
            dados["max"], 
            int(dados["num"])
        ).tolist()

    elif metodo == "passo":
        # Gera ângulos com um passo definido.
        angulos = np.arange(
            dados["min"], 
            dados["max"] + dados["passo"]/2, # Garante a inclusão do limite superior.
            dados["passo"]
        ).tolist()
    
    # Retorna a lista ordenada, sem duplicatas e com valores não negativos.
    return sorted(list(set(a for a in angulos if a >= 0)))