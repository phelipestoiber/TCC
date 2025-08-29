import numpy as np
import pandas as pd
import itertools
import time
import concurrent.futures
from scipy.optimize import fsolve, brentq
from scipy.interpolate import interp1d
from typing import Callable, Tuple, List
from src.core.ch import InterpoladorCasco
from src.utils.integration import integrar
from scipy.integrate import trapezoid

class CalculadoraEstabilidade:
    """
    Encapsula os cálculos para as curvas cruzadas de estabilidade.

    Esta classe utiliza um modelo geométrico do casco (InterpoladorCasco)
    para calcular as propriedades de estabilidade da embarcação para
    diferentes ângulos de inclinação e deslocamentos.
    """
    def __init__(self, interpolador_casco: InterpoladorCasco):
        """
        Inicializa a calculadora de estabilidade.

        Args:
            interpolador_casco (InterpoladorCasco): O objeto que contém a
                                                    geometria funcional do casco.
        """
        self.casco = interpolador_casco
        # Gera e armazena os interpoladores de bombordo (espelhados)
        self.funcoes_baliza_bb = self._gerar_interpoladores_bombordo()

    def _gerar_interpoladores_bombordo(self) -> dict:
        """
        Cria interpoladores para bombordo (BB) espelhando os de boreste (BE).

        Pressupõe simetria do casco em relação ao plano de centro.
        Para cada função de BE y=f(z), cria uma função de BB y=-f(z).
        """
        funcoes_bb = {}
        for x_pos, interpolador_be in self.casco.funcoes_baliza.items():
            # A nova função retorna o negativo do valor do interpolador original.
            funcoes_bb[x_pos] = lambda z, interp=interpolador_be: -interp(z)
        return funcoes_bb

    def _encontrar_raizes(
        self,
        funcao: Callable,
        limite_inferior: float,
        limite_superior: float,
        num_pontos: int = 101
    ) -> List[float]:
        """
        Encontra todas as raízes de uma função num dado intervalo com alta precisão.

        A metodologia é um processo robusto de duas etapas:
        1. Amostragem: O intervalo [limite_inferior, limite_superior] é dividido
           em `num_pontos`, e a função é avaliada em cada ponto para encontrar
           "brackets" (intervalos [x_i, x_{i+1}]) onde o sinal da função muda,
           indicando a presença de uma raiz.
        2. Refinamento: Para cada "bracket" encontrado, o solver `brentq` é
           utilizado para encontrar a raiz com alta precisão nesse pequeno intervalo.

        Esta abordagem é mais fiável do que usar um solver global, pois garante
        que múltiplas raízes não sejam perdidas.

        Args:
            funcao (Callable): A função a ser analisada, da forma f(x) = 0.
            limite_inferior (float): O limite inferior do intervalo de busca.
            limite_superior (float): O limite superior do intervalo de busca.
            num_pontos (int, optional): O número de pontos para a amostragem inicial.
                                         O padrão é 101.

        Returns:
            List[float]: Uma lista ordenada e sem duplicatas das raízes encontradas.
        """
        # 1. Amostragem da função no intervalo especificado
        pontos_x = np.linspace(limite_inferior, limite_superior, num_pontos)
        valores_f = funcao(pontos_x)
        raizes = []

        # 2. Itera sobre os intervalos de amostragem para encontrar mudanças de sinal
        for i in range(len(pontos_x) - 1):
            # Uma mudança de sinal entre f(x_i) e f(x_{i+1}) indica uma raiz no intervalo
            if np.sign(valores_f[i]) != np.sign(valores_f[i+1]):
                try:
                    # 3. Usa um solver de alta precisão no intervalo encontrado
                    raiz = brentq(funcao, pontos_x[i], pontos_x[i+1])
                    raizes.append(raiz)
                except (ValueError, RuntimeError):
                    # Ignora o intervalo se o brentq falhar (e.g., se um dos valores for NaN)
                    pass
        
        # 4. Retorna uma lista limpa, ordenada e sem duplicatas
        return sorted(list(set(raizes)))
    
    def _calcular_propriedades_bombordo(
        self,
        interpolador_bb: Callable,
        limites_z_baliza: np.ndarray,
        zc: float,
        theta_graus: float,
    ) -> Tuple[float, float, float]:
        """
        Calcula a área e os momentos estáticos da secção submersa de um bordo.

        A geometria da área submersa neste bordo é analisada encontrando primeiro
        a interseção entre a linha de água inclinada e o contorno do casco.
        A geometria da área submersa neste bordo é a soma de duas partes:
        1. A área sob a curva do casco, desde a quilha até a interseção com a
        linha d'água inclinada.
        2. A área do triângulo formado pela linha d'água, desde a interseção
        até o plano de centro.
        A função lida com diferentes casos, como inclinação nula ou o casco
        totalmente submerso/emerso.

        Args:
            interpolador_bb (Callable): Interpolador para o bordo (BE ou BB).
            limites_z_baliza (np.ndarray): Vetor com os limites verticais da baliza.
            zc (float): Altura da linha de água no plano de centro (y=0) [m].
            theta_graus (float): Ângulo de adernamento em graus.

        Returns:
            Tuple[float, float, float]: Tupla contendo (área [m²], momento_y [m³], momento_z [m³]).
        """
        # 1. Definição dos Limites e Parâmetros Iniciais
        z_min, z_max = limites_z_baliza.min(), limites_z_baliza.max()
        
        # O sinal do ângulo para a equação da linha de água depende do bordo.
        tan_theta = np.tan(np.deg2rad(-theta_graus))

        # 2. Caso Especial: Inclinação Nula ou inclinação Muito Grande (90 graus)
        # Cálculo simplificado, sem necessidade de encontrar raízes.
        if abs(tan_theta) < 1e-6 or abs(tan_theta) > 1e6:
            if abs(tan_theta) < 1e6:
                # Linha de água horizontal, integração até o calado zc ou z_max
                z_limite_superior = min(zc, z_max)
                area = integrar(interpolador_bb,
                                        z_min=z_min,
                                        z_max=z_limite_superior)
                mz = integrar(lambda z: z * interpolador_bb(z),
                                        z_min=z_min,
                                        z_max=z_limite_superior)
                my = integrar(lambda z: 0.5 * interpolador_bb(z)**2,
                                        z_min=z_min,
                                        z_max=z_limite_superior)
                return area, my, mz
            
            else:
                # Tangente quase infinita (90 graus), integração até z_max
                area = integrar(interpolador_bb,
                                        z_min=z_min,
                                        z_max=z_max)
                mz = integrar(lambda z: z * interpolador_bb(z),
                                        z_min=z_min,
                                        z_max=z_max)
                my = integrar(lambda z: 0.5 * interpolador_bb(z)**2,
                                        z_min=z_min,
                                        z_max=z_max)
                return area, my, mz

        # Equação da linha de água inclinada: y_wl(z)
        y_wl = lambda z: (z - zc) / tan_theta if abs(tan_theta) > 1e-6 else np.full_like(z, 1e9)
        
        # Função de diferença para encontrar a interseção casco-água
        funcao_dif = lambda z: interpolador_bb(z) - y_wl(z)

        # 3. Encontrar Ponto(s) de Interseção
        raizes = self._encontrar_raizes(funcao_dif, z_min, z_max)
        
        # Se a linha de água no centro está abaixo da quilha, o BB do navio esta fora da água,
        # a área submersa é zero.
        if zc <= z_min and not raizes:
            return 0.0, 0.0, 0.0
        
        # 4. Cálculo Baseado na Existência de Interseções
        if not raizes:
            # Se não há raízes, e zc esta acima da quilha o BB do casco está totalmente imerso.            
            # Caso contrário, está totalmente submerso. Integra-se toda a área da secção.
            area = integrar(interpolador_bb, z_min, z_max)
            mz = integrar(lambda z: z * interpolador_bb(z), z_min, z_max)
            my = integrar(lambda z: 0.5 * interpolador_bb(z)**2, z_min, z_max)
            return area, my, mz
        
        else:
            # Caso geral: há uma interseção (raiz).
            raiz = raizes[0]

            # Define os integrandos para a parte sob a baliza
            integrando_baliza_area = lambda z: interpolador_bb(z)
            integrando_baliza_mz = lambda z: z * interpolador_bb(z)
            integrando_baliza_my = lambda z: 0.5 * interpolador_bb(z)**2

            # Define os integrandos para a parte sob a linha de água (forma de cunha)
            integrando_wl_area = lambda z: y_wl(z)
            integrando_wl_mz = lambda z: z * y_wl(z)
            integrando_wl_my = lambda z: 0.5 * y_wl(z)**2
            
            # A integração é dividida em duas partes, na interseção 'raiz'.
            # Parte 1: Da quilha até à raiz, segue o contorno do casco.
            area1 = integrar(integrando_baliza_area, z_min, raiz)
            mz1 = integrar(integrando_baliza_mz, z_min, raiz)
            my1 = integrar(integrando_baliza_my, z_min, raiz)

            # Parte 2: Da raiz até à linha de água no centro (zc), segue a linha de água.
            # O limite superior da integração é zc ou z_max.
            z_limite_superior_wl = min(zc, z_max)

            area2 = integrar(integrando_wl_area, raiz, z_limite_superior_wl)
            mz2 = integrar(integrando_wl_mz, raiz, z_limite_superior_wl)
            my2 = integrar(integrando_wl_my, raiz, z_limite_superior_wl)
            
            # Soma as duas partes para obter o total do bordo.
            area_total = area1 + area2
            momento_z_total = mz1 + mz2
            momento_y_total = my1 + my2

            return area_total, momento_y_total, momento_z_total
        
    def _calcular_propriedades_boreste(
        self,
        interpolador_be: Callable,
        limites_z_baliza: np.ndarray,
        zc: float,
        theta_graus: float
    ) -> Tuple[float, float, float]:
        """
        Calcula área e momentos estáticos da secção submersa de boreste (bordo emerso).

        A geometria da área submersa neste bordo é a área entre a curva do casco e
        a linha de água inclinada. A função calcula essa área integrando a diferença
        y(z) = y_casco(z) - y_agua(z) nos intervalos verticais corretos, que são
        determinados pelos pontos de interseção (raízes) entre as duas curvas.

        Args:
            interpolador_be (Callable): Interpolador para o bordo de boreste (y >= 0).
            limites_z_baliza (np.ndarray): Vetor de coordenadas Z que define os limites da baliza.
            zc (float): Altura da linha de água no plano de centro (y=0) [m].
            theta_graus (float): Ângulo de adernamento em graus (positivo para boreste).

        Returns:
            Tuple[float, float, float]: Tupla contendo (área [m²], momento_y [m³], momento_z [m³]).
        """
        # 1. Definição dos Limites e Parâmetros
        z_min, z_max = limites_z_baliza.min(), limites_z_baliza.max()

        # O sinal do ângulo para a equação da linha de água depende do bordo.
        tan_theta = np.tan(np.deg2rad(-theta_graus))

        # 2. Caso Especial: Inclinação Nula ou inclinação Muito Grande (90 graus)
        # Cálculo simplificado, sem necessidade de encontrar raízes.
        if abs(tan_theta) < 1e-6 or abs(tan_theta) > 1e6:
            if abs(tan_theta) < 1e6:
                # Linha de água horizontal, integração até o calado zc ou z_max
                z_limite_superior = min(zc, z_max)
                area = integrar(interpolador_be,
                                        z_min=z_min,
                                        z_max=z_limite_superior)
                mz = integrar(lambda z: z * interpolador_be(z),
                                        z_min=z_min,
                                        z_max=z_limite_superior)
                my = integrar(lambda z: 0.5 * interpolador_be(z)**2,
                                        z_min=z_min,
                                        z_max=z_limite_superior)
                return area, my, mz
            
            else:
                # Tangente quase infinita (90 graus), integração até z_max
                area = integrar(interpolador_be,
                                        z_min=z_min,
                                        z_max=z_max)
                mz = integrar(lambda z: z * interpolador_be(z),
                                        z_min=z_min,
                                        z_max=z_max)
                my = integrar(lambda z: 0.5 * interpolador_be(z)**2,
                                        z_min=z_min,
                                        z_max=z_max)
                return area, my, mz


        # Equação da linha de água inclinada: y_wl(z)
        y_wl = lambda z: (zc - z) / tan_theta

        # Função de diferença para encontrar a interseção casco-água
        funcao_dif = lambda z: interpolador_be(z) - y_wl(z)

        # 3. Encontrar Ponto(s) de Interseção
        raizes = self._encontrar_raizes(funcao_dif, z_min, z_max)
        
        # 4. Definição dos Integrandos
        integrando_baliza_area = interpolador_be
        integrando_baliza_mz = lambda z: z * interpolador_be(z)
        integrando_baliza_my = lambda z: 0.5 * interpolador_be(z)**2
        
        integrando_wl_area = y_wl
        integrando_wl_mz = lambda z: z * y_wl(z)
        integrando_wl_my = lambda z: 0.5 * y_wl(z)**2

        # 5. Lógica de Integração Baseada em zc e raízes
        # Caso 1: Linha de água no centro (zc) abaixo da quilha
        if zc < z_min:
            # Se há duas raízes, significa que a linha d'água cruza o casco duas vezes (geralmente em baixos e médios ângulos)
            if len(raizes) >= 2:
                r1, r2 = raizes[0], raizes[1]
                area1 = integrar(integrando_baliza_area, r1, r2)
                mz1 = integrar(integrando_baliza_mz, r1, r2)
                my1 = integrar(integrando_baliza_my, r1, r2)
                
                area2 = integrar(integrando_wl_area, r1, r2)
                mz2 = integrar(integrando_wl_mz, r1, r2)
                my2 = integrar(integrando_wl_my, r1, r2)
                return area1 - area2, my1 - my2, mz1 - mz2
            
            elif len(raizes) == 1:
                # Se há uma raíz, significa que a linha d'água cruza o casco uma vez e outra no convés (geralmente em grandes ângulos)
                r1, r2 = raizes[0], z_max
                area1 = integrar(integrando_baliza_area, r1, r2)
                mz1 = integrar(integrando_baliza_mz, r1, r2)
                my1 = integrar(integrando_baliza_my, r1, r2)
                
                area2 = integrar(integrando_wl_area, r1, r2)
                mz2 = integrar(integrando_wl_mz, r1, r2)
                my2 = integrar(integrando_wl_my, r1, r2)
                return area1 - area2, my1 - my2, mz1 - mz2

            else: # Se não há interseção, significa que o casco está totalmente emerso
                return 0.0, 0.0, 0.0
            
        # Caso 2: Linha de água no centro entre a quilha e convés
        elif z_min <= zc < z_max:
            if len(raizes) == 1: # Caso tradicional de um bordo a emergir
                raiz = raizes[0]

                area1 = integrar(integrando_baliza_area, z_min, raiz)
                mz1 = integrar(integrando_baliza_mz, z_min, raiz)
                my1 = integrar(integrando_baliza_my, z_min, raiz)
                
                # A integral da cunha de água é de zc até à raiz
                area2 = integrar(integrando_wl_area, zc, raiz)
                mz2 = integrar(integrando_wl_mz, zc, raiz)
                my2 = integrar(integrando_wl_my, zc, raiz)
                return area1 - area2, my1 - my2, mz1 - mz2
            
            else: # Nenhuma raiz (interseção com o convés)
                area1 = integrar(integrando_baliza_area, z_min, z_max)
                mz1 = integrar(integrando_baliza_mz, z_min, z_max)
                my1 = integrar(integrando_baliza_my, z_min, z_max)
                
                area2 = integrar(integrando_wl_area, zc, z_max)
                mz2 = integrar(integrando_wl_mz, zc, z_max)
                my2 = integrar(integrando_wl_my, zc, z_max)
                return area1 - area2, my1 - my2, mz1 - mz2
        
        # Caso 3: Linha de água no centro acima do convés (casco totalmente submerso)
        else:
            area = integrar(integrando_baliza_area, z_min, z_max)
            mz = integrar(integrando_baliza_mz, z_min, z_max)
            my = integrar(integrando_baliza_my, z_min, z_max)
            return area, my, mz

    def _calcular_propriedades_secao_inclinada(
        self,
        x_pos: float,
        zc: float,
        theta_graus: float
    ) -> Tuple[float, float, float]:
        """
        Calcula a área e os momentos estáticos totais de uma única secção transversal inclinada.

        Este método utiliza os cálculos de cada bordo (bombordo e boreste) para
        determinar as propriedades completas da secção submersa.

        Args:
            x_pos (float): A posição longitudinal (x) da secção.
            zc (float): A altura da linha de água no plano de centro (y=0) [m].
            theta_graus (float): O Ângulo de adernamento em graus [°].

        Returns:
            Tuple[float, float, float]: A área, momento em Y e momento em Z da secção.
        """
        # 1. Obter os interpoladores para a secção atual
        interp_be = self.casco.funcoes_baliza.get(x_pos)
        interp_bb = self.funcoes_baliza_bb.get(x_pos)

        # Se a secção não tiver um interpolador válido, retorna zero.
        if not interp_be or not interp_bb:
            return 0.0, 0.0, 0.0

        # 2. Definir os limites verticais da baliza para a integração
        # Usamos os pontos originais do interpolador de boreste para definir os limites.
        limites_z = interp_be.x 

        # 3. Calcular as propriedades para cada bordo separadamente
        area_bb, my_bb, mz_bb = self._calcular_propriedades_bombordo(
            interp_bb, limites_z, zc, theta_graus
        )
        area_be, my_be, mz_be = self._calcular_propriedades_boreste(
            interp_be, limites_z, zc, theta_graus
        )
        
        # 4. Somar os resultados para obter o total da secção
        area_total = area_bb + area_be
        momento_y_total = my_bb + my_be
        momento_z_total = mz_bb + mz_be

        return area_total, momento_y_total, momento_z_total

    # --- MÉTODO PRINCIPAL ADAPTADO ---
    def _calcular_propriedades_para_zc_inclinado(
        self, zc: float, theta_graus: float
    ) -> Tuple[float, float, float]:
        """
        Calcula as propriedades da carena inclinada para uma dada linha de água.

        Para uma linha de água definida por (zc, theta_graus), esta função itera sobre
        todas as balizas e calcula a área e momentos de cada secção submersa.
        Finalmente, integra essas propriedades ao longo do comprimento para
        obter o volume total e os momentos de volume da carena.

        Args:
            zc (float): A altura da linha de água no plano de centro (y=0) [m].
            theta_graus (float): O ângulo de adernamento em graus [°].

        Returns:
            Tuple[float, float, float]: Tupla com o volume [m³], momento de volume em Y [m⁴],
                                      e momento de volume em Z [m⁴].
        """
        # 1. Inicializar listas para armazenar as propriedades de cada secção
        areas_secao = []
        momentos_y_secao = []
        momentos_z_secao = []
        
        # As posições X das balizas, que servirão como eixo de integração longitudinal.
        coords_x = self.casco.posicoes_balizas

        # 2. Iterar sobre cada baliza para calcular as suas propriedades submersas
        for x_pos in coords_x:
            area_s, my_s, mz_s = self._calcular_propriedades_secao_inclinada(
                x_pos, zc, theta_graus
            )
            areas_secao.append(area_s)
            momentos_y_secao.append(my_s)
            momentos_z_secao.append(mz_s)

        # 3. Integrar as propriedades das secções ao longo do comprimento do navio
        # Usa a regra do trapézio para integrar a curva de áreas e as curvas de momentos.
        volume_total = trapezoid(y=areas_secao, x=coords_x)
        momento_y_total = trapezoid(y=momentos_y_secao, x=coords_x)
        momento_z_total = trapezoid(y=momentos_z_secao, x=coords_x)
        
        return volume_total, momento_y_total, momento_z_total
    
    def _encontrar_zc_para_volume(
        self,
        volume_desejado: float,
        theta_graus: float,
        zc_chute_inicial: float,
        boca_maxima: float, # Parâmetro necessário para o fallback do brentq
        pontal_maximo: float # Parâmetro necessário para o fallback do brentq
    ) -> Tuple[float, float, float]:
        """
        Encontra a linha de água (zc) que corresponde a um volume de carena desejado.

        Utiliza um solver numérico (fsolve) para encontrar a altura da linha de água
        no centro (zc) que resulta no volume de carena especificado, para um dado
        ângulo de inclinação. A função a ser zerada pelo solver é a diferença entre o
        volume calculado e o volume desejado.

        Args:
            volume_desejado (float): O volume de carena alvo [m³].
            theta_graus (float): O ângulo de adernamento em graus [°].
            zc_chute_inicial (float): Um valor inicial para zc para ajudar o solver.
            boca_maxima (float): A boca máxima da embarcação [m].
            pontal_maximo (float): O pontal máximo da embarcação [m].

        Returns:
            Tuple[float, float, float]: Tupla com zc da solução, centroide Y (Y_cb),
                                      e centroide Z (Z_cb) da carena.
        """
        # 1. Definição da Função Objetivo para o Solver
        # O solver precisa de uma função que receba 'zc' e retorne o erro de volume.
        # O [0] pega apenas o 'volume' retornado por _calcular_propriedades_para_zc_inclinado.
        funcao_objetivo = lambda zc_input: self._calcular_propriedades_para_zc_inclinado(
            zc=float(zc_input),
            theta_graus=theta_graus
        )[0] - volume_desejado

        # 2. Execução do Solver Numérico
        try:
            # fsolve tenta encontrar o valor de 'zc' que torna 'funcao_objetivo' igual a zero.
            zc_solucao = fsolve(funcao_objetivo, x0=zc_chute_inicial)[0]
            print(f"INFO: Solver encontrou zc = {zc_solucao:.4f} m para theta = {theta_graus:.1f}°")
        
        except Exception:
            # Se fsolve falhar, tenta um método mais robusto (brentq) com um intervalo definido.
            print(f"AVISO: fsolve falhou. Tentando com brentq para theta = {theta_graus:.1f}°...")
            try:
                tan_theta_abs = abs(np.tan(np.deg2rad(theta_graus)))
                # Define um intervalo de busca para zc que realisticamente contém a solução.
                zc_min_busca = -0.5 * boca_maxima * tan_theta_abs - 1.0 # Adiciona margem
                zc_max_busca = 0.5 * boca_maxima * tan_theta_abs + pontal_maximo + 1.0 # Adiciona margem
                
                zc_solucao = brentq(funcao_objetivo, a=zc_min_busca, b=zc_max_busca)
                print(f"INFO: brentq encontrou zc = {zc_solucao:.4f} m")

            except Exception as e:
                # Se ambos os solvers falharem, a condição é provavelmente instável.
                print(f"ERRO: Não foi possível encontrar uma linha de água de equilíbrio para theta = {theta_graus:.1f}°.")
                print("      Isso pode indicar que o convés submerge ou uma condição instável.")
                print(f"      Erro do solver: {e}")
                return float('nan'), float('nan'), float('nan')

        # 3. Com o zc correto, calcular as propriedades finais
        volume_final, momento_y_total, momento_z_total = self._calcular_propriedades_para_zc_inclinado(
            zc=zc_solucao, theta_graus=theta_graus
        )

        # 4. Calcular os centroides da carena (Y_cb, Z_cb)
        # O centroide é o momento de volume dividido pelo volume.
        y_cb = momento_y_total / volume_final if volume_final > 1e-6 else 0.0
        z_cb = momento_z_total / volume_final if volume_final > 1e-6 else 0.0

        return zc_solucao, y_cb, z_cb

    def calcular_kn_para_ponto(
        self,
        deslocamento: float,
        angulo_graus: float,
        densidade_agua: float,
        interp_chute_zc: Callable,
        boca_maxima: float,
        pontal_maximo: float
    ) -> float:
        """
        Calcula um único valor do braço de endireitamento (KN) para um deslocamento e ângulo específicos.

        Esta função é o núcleo do cálculo de estabilidade para um ponto. Ela:
        1. Converte o deslocamento e o ângulo para as unidades necessárias.
        2. Usa um interpolador pré-calculado para obter um chute inicial de 'zc'.
        3. Chama '_encontrar_zc_para_volume' para achar a linha de água de equilíbrio.
        4. Usa a fórmula KN = Y_cb * cos(θ) + Z_cb * sin(θ) para calcular o braço de endireitamento.

        Args:
            deslocamento (float): O deslocamento para o qual o KN será calculado [t].
            angulo_graus (float): O ângulo de adernamento [°].
            densidade_agua (float): A densidade da água [t/m³].
            interp_chute_zc (Callable): Uma função que recebe um deslocamento [t]
                                      e retorna um calado [m] para ser usado como chute inicial.
            boca_maxima (float): A boca máxima da embarcação [m].
            pontal_maximo (float): O pontal máximo da embarcação [m].

        Returns:
            float: O valor do braço de endireitamento KN [m]. Retorna 'nan' se o cálculo falhar.
        """
        # 1. Preparação dos dados para o cálculo do ponto
        volume_desejado = deslocamento / densidade_agua
        angulo_rad = np.deg2rad(angulo_graus)
        
        # Se o volume for desprezível, o KN é zero.
        if volume_desejado < 1e-3:
            return 0.0
        
        # 2. Obtenção do chute inicial para 'zc' a partir das curvas hidrostáticas
        zc_chute = float(interp_chute_zc(deslocamento))

        # 3. Encontrar a linha de água de equilíbrio e os centroides
        zc_final, y_cb, z_cb = self._encontrar_zc_para_volume(
            volume_desejado=volume_desejado,
            theta_graus=angulo_graus,
            zc_chute_inicial=zc_chute,
            boca_maxima=boca_maxima,
            pontal_maximo=pontal_maximo
        )
        
        # Se o solver falhar e retornar NaN, propaga o NaN para o resultado.
        if np.isnan(zc_final):
            return np.nan

        # 4. Cálculo do Braço de Endireitamento (KN)
        # A fórmula projeta o vetor do centro de carena (Y_cb, Z_cb) na direção
        # perpendicular à linha de flutuação inclinada.
        kn_valor = y_cb * np.cos(angulo_rad) + z_cb * np.sin(angulo_rad)
        
        return kn_valor
    
def calcular_kn_worker(args):
    """Função worker para o cálculo paralelo de um ponto KN."""
    (desloc, angulo, casco_interpolado, densidade, 
     interp_chute_zc, boca, pontal) = args
    
    # Cria uma instância da calculadora dentro do processo worker
    calculadora_kn = CalculadoraEstabilidade(casco_interpolado)
    
    kn = calculadora_kn.calcular_kn_para_ponto(
        deslocamento=desloc, angulo_graus=angulo,
        densidade_agua=densidade, interp_chute_zc=interp_chute_zc,
        boca_maxima=boca, pontal_maximo=pontal
    )
    return desloc, angulo, kn

class CalculadoraCurvasCruzadas:
    """
    Orquestra o cálculo completo das curvas cruzadas de estabilidade (KN).
    """
    def __init__(self, casco_interpolado: InterpoladorCasco, df_hidrostatico: pd.DataFrame, dados_embarcacao: dict):
        self.casco = casco_interpolado
        self.df_hidrostatico = df_hidrostatico
        self.dados_embarcacao = dados_embarcacao

    def calcular_curvas_kn(self, lista_deslocamentos: list, lista_angulos: list) -> pd.DataFrame:
        """Executa o cálculo de todos os pontos KN em paralelo."""
        print("\n--- INICIANDO CÁLCULO DAS CURVAS CRUZADAS ---")
        start_time = time.perf_counter()

        # Criar interpolador para o chute inicial de zc (Calado = f(Deslocamento))
        df_hidro_unico = self.df_hidrostatico.drop_duplicates(subset=['Desloc. (t)'])
        interp_chute_zc = interp1d(
            df_hidro_unico['Desloc. (t)'], df_hidro_unico['Calado (m)'],
            bounds_error=False, fill_value="extrapolate"
        )

        # Preparar tarefas para o multiprocessing
        tarefas = list(itertools.product(
            lista_deslocamentos, lista_angulos, 
            [self.casco], [float(self.dados_embarcacao['densidade'])], 
            [interp_chute_zc], [float(self.dados_embarcacao['boca'])], 
            [float(self.dados_embarcacao['pontal'])]
        ))

        resultados_kn = []
        with concurrent.futures.ProcessPoolExecutor() as executor:
            resultados_kn = list(executor.map(calcular_kn_worker, tarefas))

        duration = time.perf_counter() - start_time
        print(f"-> Cálculo de {len(tarefas)} pontos KN finalizado em {duration:.2f} segundos.")

        # Processar e retornar resultados
        df_kn_flat = pd.DataFrame(resultados_kn, columns=['Deslocamento', 'Ângulo', 'KN'])
        df_kn_pivot = df_kn_flat.pivot(index='Deslocamento', columns='Ângulo', values='KN').reset_index()
        df_kn_pivot = df_kn_pivot.rename(columns={'index': 'Desloc. (t)'})
        
        return df_kn_pivot
    
    