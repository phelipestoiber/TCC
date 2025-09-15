import numpy as np
from scipy.integrate import trapezoid
from typing import Callable

def integrar(
    funcao_a_integrar: Callable,
    limite_inferior: float,
    limite_superior: float,
    passo: float = 0.001
) -> float:
    """
    Calcula a integral definida de uma função utilizando a regra do trapézio.

    Este método gera uma série de pontos entre os limites de integração,
    avalia a função em cada um desses pontos e, em seguida, utiliza a regra
    do trapézio para aproximar a área sob a curva.

    Args:
        funcao_a_integrar (Callable): A função a ser integrada. Deve aceitar
                                      um array numpy de pontos como entrada.
        limite_inferior (float): O limite inferior da integração.
        limite_superior (float): O limite superior da integração.
        passo (float, optional): A distância entre os pontos de avaliação.
                                 Um passo menor aumenta a precisão, mas também
                                 o tempo de cálculo. O padrão é 0.01.

    Returns:
        float: O resultado numérico da integral.
    """
    # Se o intervalo de integração for inválido ou muito pequeno para ter
    # pelo menos dois pontos, a área é, por definição, zero.
    if limite_inferior >= limite_superior:
        return 0.0

    # Gera um array de pontos de avaliação desde o limite inferior até o superior.
    # Adiciona-se um pequeno valor ao limite superior para garantir que ele seja incluído.
    pontos_x = np.arange(limite_inferior, limite_superior + passo / 2, passo)

    # Segunda verificação para garantir que tem-se ao menos 2 pontos
    # para a regra do trapézio funcionar.
    if pontos_x.size < 2:
        return 0.0
    
    # Avalia a função em todos os pontos gerados.
    pontos_y = funcao_a_integrar(pontos_x)
    
    # A regra do trapézio requer que valores NaN (Not a Number) sejam tratados.
    # Substituímos quaisquer NaNs por 0.0 para não afetar a soma da área.
    pontos_y = np.nan_to_num(pontos_y, nan=0.0)

    # Calcula a área utilizando a função trapezoid da SciPy, que implementa
    # a regra numérica de forma eficiente.
    area = trapezoid(y=pontos_y, x=pontos_x)

    return area