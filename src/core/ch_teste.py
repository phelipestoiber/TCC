# -*- coding: utf-8 -*-
"""
Laboratório para Teste e Desenvolvimento de Funções Hidrostáticas.

Este módulo autossuficiente serve como um ambiente de desenvolvimento para
refatorar e testar a lógica de cálculo de propriedades hidrostáticas.

Ele contém todas as classes e funções necessárias para definir uma geometria de
casco e calcular suas propriedades hidrostáticas de forma modular.

Componentes Incluídos:
- integrar: Função de integração numérica pela regra do trapézio.
- InterpoladorCasco: Classe que representa a geometria do casco e encapsula a
  lógica de interpolação, incluindo o método 'obter_meia_boca'.
- aplicar_correcao_deflexao: Função para simular a deflexão do casco.
- PropriedadesHidrostaticasBase: A classe "upper level" com a lógica principal
  para os cálculos hidrostáticos.
"""

import pandas as pd
import numpy as np
from scipy.interpolate import interp1d, PchipInterpolator, make_interp_spline
from scipy.integrate import trapezoid
from scipy.optimize import fsolve
from typing import Dict, Any, Tuple, Callable, List
import copy
import warnings

# =============================================================================
# 1. FUNÇÕES AUXILIARES (UTILITIES)
# Copiado de 'src/utils/integration.py' para tornar este arquivo autônomo.
# =============================================================================
def integrar(
    funcao_a_integrar: Callable,
    limite_inferior: float,
    limite_superior: float,
    passo: float = 0.01
) -> float:
    """
    Calcula a integral definida de uma função utilizando a regra do trapézio.
    """
    if limite_inferior >= limite_superior:
        return 0.0
    pontos_x = np.arange(limite_inferior, limite_superior + passo / 2, passo)
    if pontos_x.size < 2:
        return 0.0
    pontos_y = funcao_a_integrar(pontos_x)
    pontos_y = np.nan_to_num(pontos_y)
    return trapezoid(pontos_y, pontos_x)

# =============================================================================
# 2. CLASSE DE GEOMETRIA DO CASCO
# Definição completa da classe que representa o casco.
# =============================================================================
class InterpoladorCasco:
    """
    Representa a geometria do casco através de funções de interpolação.

    Esta classe encapsula a lógica de transformação de uma tabela de cotas
    discreta em um modelo geométrico contínuo.
    """
    def __init__(self, tabela_cotas: pd.DataFrame, metodo_interp: str = 'linear'):
        self.tabela_cotas = tabela_cotas
        self.metodo_interp = metodo_interp
        self.z_min = tabela_cotas['z'].min()
        self.z_max = tabela_cotas['z'].max()
        self.x_min = tabela_cotas['x'].min()
        self.x_max = tabela_cotas['x'].max()
        self.lpp = self.x_max - self.x_min
        self.z_min_balizas = self.tabela_cotas.groupby('x')['z'].min().to_dict()
        self.funcoes_baliza = self._criar_interpoladores_balizas()
        self.funcao_perfil = self._criar_interpolador_perfil()

    def _criar_interpoladores_balizas(self) -> Dict[float, PchipInterpolator]:
        interpoladores = {}
        for x_val, grupo in self.tabela_cotas.groupby('x'):
            df_baliza = grupo.sort_values('z')
            z_coords = df_baliza['z'].values
            y_coords = df_baliza['y'].values

            if len(z_coords) > 1:
                if self.metodo_interp == 'pchip':
                    interpolador = PchipInterpolator(z_coords, y_coords, extrapolate=False)
                else: # Padrão para 'linear'
                    # interpolador = make_interp_spline(z_coords, y_coords, k=1)
                    interpolador = interp1d(z_coords, y_coords, kind='linear', bounds_error=False, fill_value=0.0)
                interpoladores[x_val] = interpolador
        return interpoladores

    def _criar_interpolador_perfil(self) -> PchipInterpolator:
        perfil_df = self.tabela_cotas.loc[self.tabela_cotas.groupby('x')['z'].idxmax()]
        perfil_df = perfil_df.sort_values(by='x')
        x_coords = perfil_df['x'].values
        z_coords = perfil_df['z'].values

        if len(x_coords) > 1:
            if self.metodo_interp == 'pchip':
                return PchipInterpolator(x_coords, z_coords, extrapolate=False)
            else: # Padrão para 'linear'
                # return make_interp_spline(x_coords, z_coords, k=1)
                return interp1d(x_coords, z_coords, kind='linear', bounds_error=False, fill_value=0.0)

    def obter_meia_boca(self, x: float, z: float) -> float:
        """
        Obtém a meia-boca (y) para uma dada estação 'x' e altura 'z'.
        O metodo verifica se 'x' é um array e aplica a lógica para cada elemento.

        Este método é a interface pública para consultar a geometria do casco.
        Ele encapsula toda a lógica de interpolação.

        Args:
            x (float): A coordenada longitudinal da estação a ser consultada.
            z (float): A coordenada vertical na qual a meia-boca é desejada.

        Returns:
            float: O valor da meia-boca (y). Retorna 0 se o ponto (x, z) estiver
                   fora dos limites da geometria definida.
        """

        # Se 'x' for um array, aplica a lógica para cada elemento.
        if isinstance(x, np.ndarray):
            # Usamos np.vectorize para criar uma versão da função que opera
            # em arrays de forma eficiente.
            v_meia_boca = np.vectorize(self.obter_meia_boca, otypes=[float])
            return v_meia_boca(x, z)

        funcao_interpoladora = self.funcoes_baliza.get(x)
        if funcao_interpoladora:
            meia_boca = funcao_interpoladora(z)
            return np.nan_to_num(meia_boca)
        else:
            if isinstance(z, np.ndarray):
                return np.zeros_like(z)
            return 0.0

# =============================================================================
# 3. FUNÇÃO DE CORREÇÃO DE DEFLEXÃO
# =============================================================================
def aplicar_correcao_deflexao(
    casco_original: InterpoladorCasco,
    deflexao_re: float,
    deflexao_meio: float,
    deflexao_vante: float,
    lpp: float
) -> InterpoladorCasco:
    """
    Aplica uma correção de deflexão (hogging/sagging) à tabela de cotas.
    """
    x_pontos = np.array([0, lpp / 2, lpp])
    y_pontos = np.array([deflexao_re, deflexao_meio, deflexao_vante])
    funcao_deflexao = interp1d(x_pontos, y_pontos, kind='quadratic', fill_value="extrapolate")
    tabela_cotas_corrigida = copy.deepcopy(casco_original.tabela_cotas)

    for x_pos in tabela_cotas_corrigida['x'].unique():
        correcao_z = funcao_deflexao(x_pos)
        indices_baliza = tabela_cotas_corrigida['x'] == x_pos
        tabela_cotas_corrigida.loc[indices_baliza, 'z'] += correcao_z
        
    casco_corrigido = InterpoladorCasco(tabela_cotas_corrigida, metodo_interp=casco_original.metodo_interp)
    return casco_corrigido

# =============================================================================
# 4. CLASSE BASE PARA CÁLCULOS HIDROSTÁTICOS
# =============================================================================
class PropriedadesHidrostaticasBase:
    """
    Classe base para o cálculo de propriedades hidrostáticas.

    Esta classe "upper level" contém os métodos de cálculo que são comuns
    a qualquer condição de flutuação, seja em carena direita ou trimada.
    A lógica principal, como o cálculo de volume, centros e momentos de inércia,
    é definida aqui.

    As classes filhas serão responsáveis por implementar os métodos de "baixo nível"
    que calculam as áreas e momentos das seções transversais e longitudinais,
    pois é nesses cálculos que a diferença entre carena direita e trimada se manifesta.
    """
    def __init__(self, casco: InterpoladorCasco, densidade: float = 1.025):
        """
        Inicializa a calculadora de propriedades base.

        Args:
            casco (InterpoladorCasco): O objeto que representa a geometria do casco.
            densidade (float): A densidade da água (padrão: 1.025 t/m³).
        """
        self.casco = casco
        self.densidade = densidade
        self.resultados: Dict[str, Any] = {} # Dicionário para armazenar os resultados

        self.interpolador_areas: Callable = None
        self.interpolador_momentos_verticais: Callable = None

    def _calcular_volume_deslocamento(self) -> Tuple[float, float]:
        """
        Calcula o volume submerso e o deslocamento da embarcação.

        Integra a área das seções transversais ao longo do comprimento de flutuação.
        Este método depende de '_calcular_area_secao', que deve ser implementado
        pelas classes filhas.

        Returns:
            Tuple[float, float]: O volume submerso (m³) e o deslocamento (toneladas).
        """
        l_inf, l_sup = self.resultados['Lim_Inf_Flut'], self.resultados['Lim_Sup_Flut']

        # A função a ser integrada é a área da seção transversal em cada 'x'
        # Note que '_calcular_area_secao' será diferente para carena direita e trimada
        volume = integrar(self._calcular_area_secao, l_inf, l_sup)

        deslocamento = volume * self.densidade
        return volume, deslocamento

    def _calcular_lcb(self) -> float:
        """
        Calcula a posição longitudinal do centro de carena (LCB).

        Integra o momento longitudinal das áreas de seção transversal e o divide
        pelo volume total.

        Returns:
            float: A coordenada x do centro de carena (LCB) em metros.
        """
        l_inf, l_sup = self.resultados['Lim_Inf_Flut'], self.resultados['Lim_Sup_Flut']

        # Momento longitudinal = integral da (área da seção * x) dx
        momento_longitudinal = integrar(
            lambda x: self._calcular_area_secao(x) * x,
            l_inf,
            l_sup
        )

        volume = self.resultados['Volume']
        # Evita divisão por zero se o volume for nulo
        lcb = momento_longitudinal / volume if volume else 0.0
        return lcb

    def _calcular_vcb(self) -> float:
        """
        Calcula a posição vertical do centro de carena (VCB ou KB).

        Integra o momento vertical das áreas de seção transversal e o divide
        pelo volume total.

        Returns:
            float: A coordenada z do centro de carena (VCB) em metros.
        """
        l_inf, l_sup = self.resultados['Lim_Inf_Flut'], self.resultados['Lim_Sup_Flut']
        
        # Momento vertical = integral do (momento vertical da seção) dx
        # O método '_calcular_momento_vertical_secao' deve ser implementado pela subclasse.
        momento_vertical_total = integrar(
            self._calcular_momento_vertical_secao,
            l_inf,
            l_sup
        )

        volume = self.resultados['Volume']
        vcb = momento_vertical_total / volume if volume else 0.0
        return vcb

    def _calcular_lcf(self) -> float:
        """
        Calcula a posição longitudinal do centro de flutuação (LCF).

        O LCF é o centroide da área do plano de flutuação.

        Returns:
            float: A coordenada x do centro de flutuação (LCF) em metros.
        """
        l_inf, l_sup = self.resultados['Lim_Inf_Flut'], self.resultados['Lim_Sup_Flut']
        calado_atual = self.resultados['Calado']

        # Momento longitudinal da área de flutuação = integral da (meia_boca * 2 * x) dx
        momento_area_flutuacao = integrar(
            lambda x: self.casco.obter_meia_boca(x, calado_atual) * 2 * x,
            l_inf,
            l_sup
        )

        area_flutuacao = self.resultados['AW']
        lcf = momento_area_flutuacao / area_flutuacao if area_flutuacao else 0.0
        return lcf

    def _calcular_momento_inercia_transversal(self) -> float:
        """
        Calcula o momento de inércia da área do plano de flutuação em
        relação a um eixo longitudinal que passa pelo centroide (LCF).

        Returns:
            float: Momento de inércia transversal (IT) em m^4.
        """
        l_inf, l_sup = self.resultados['Lim_Inf_Flut'], self.resultados['Lim_Sup_Flut']
        calado_atual = self.resultados['Calado']

        # IT = integral de (1/3 * (meia_boca)^3 * 2) dx
        momento_inercia_transversal = integrar(
            lambda x: (2 / 3) * (self.casco.obter_meia_boca(x, calado_atual) ** 3),
            l_inf,
            l_sup
        )
        return momento_inercia_transversal

    def _calcular_momento_inercia_longitudinal(self) -> float:
        """
        Calcula o momento de inércia da área do plano de flutuação em
        relação a um eixo transversal que passa pelo centroide (LCF).

        Returns:
            float: Momento de inércia longitudinal (IL) em m^4.
        """
        l_inf, l_sup = self.resultados['Lim_Inf_Flut'], self.resultados['Lim_Sup_Flut']
        calado_atual = self.resultados['Calado']
        lcf = self.resultados['LCF']

        # IL = integral de (boca * (x - LCF)^2) dx
        momento_inercia_longitudinal = integrar(
            lambda x: (self.casco.obter_meia_boca(x, calado_atual) * 2) * ((x - lcf) ** 2),
            l_inf,
            l_sup
        )
        return momento_inercia_longitudinal

    def _calcular_propriedades_derivadas(self) -> None:
        """
        Calcula as propriedades hidrostáticas que são derivadas de outras.
        Esta função preenche o dicionário `self.resultados` com os valores finais.
        """
        # Raio metacêntrico transversal (BMt)
        self.resultados['BMt'] = self.resultados['IT'] / self.resultados['Volume'] if self.resultados['Volume'] else 0.0
        # Raio metacêntrico longitudinal (BMl)
        self.resultados['BMl'] = self.resultados['IL'] / self.resultados['Volume'] if self.resultados['Volume'] else 0.0

        # Posição vertical do metacentro transversal (KMt)
        self.resultados['KMt'] = self.resultados['VCB'] + self.resultados['BMt']
        # Posição vertical do metacentro longitudinal (KMl)
        self.resultados['KMl'] = self.resultados['VCB'] + self.resultados['BMl']

        # Momento para trimar 1 cm (MT1cm ou MTC)
        # MT1cm = (Deslocamento * GML) / (100 * Lpp) -> GML ~= BML
        lpp = self.casco.lpp
        self.resultados['MT1cm'] = (self.resultados['Desloc'] * self.resultados['BMl']) / (100 * lpp) if lpp else 0.0
        
        # Toneladas por centímetro de imersão (TPC)
        self.resultados['TPC'] = (self.resultados['AW'] * self.densidade) / 100

    # Métodos que devem ser obrigatoriamente implementados pelas classes filhas
    # --------------------------------------------------------------------------
    # def _calcular_area_secao(self, x: float) -> float: raise NotImplementedError
    # def _calcular_momento_vertical_secao(self, x: float) -> float: raise NotImplementedError
    # def calcular_todas_propriedades(self) -> Dict[str, Any]: raise NotImplementedError
    # def _calcular_area_plano_flutuacao(self) -> float: raise NotImplementedError
    # def _calcular_dimensoes_linha_dagua(self) -> Tuple[float, float]: raise NotImplementedError

# =============================================================================
# 5. NOVA CLASSE FILHA PARA CARENA DIREITA
# =============================================================================
class PropriedadesHidrostaticasDireita(PropriedadesHidrostaticasBase):
    """
    Calcula as propriedades hidrostáticas para uma condição de flutuação
    em carena direita (sem trim ou banda).
    """
    def __init__(self, casco: InterpoladorCasco, calado: float, densidade: float = 1.025):
        """
        Inicializa a calculadora para um calado específico.

        Args:
            casco (InterpoladorCasco): O objeto de geometria do casco.
            calado (float): O calado para o qual os cálculos serão feitos.
            densidade (float): A densidade da água.
        """
        super().__init__(casco, densidade)
        self.calado = calado

    def _calcular_dimensoes_linha_dagua(self) -> Tuple[float, float]:
        """
        Encontra os limites de vante e de ré da linha d'água.
        """
        # --- Cálculo da Boca na Linha d'Água (BWL) ---
        if self.calado >= self.casco.z_max:
            return 0.0, 0.0
        else:
            meia_boca_max = 0.0
            for funcao_baliza in self.casco.funcoes_baliza.values():
                meia_boca_atual = np.nan_to_num(float(funcao_baliza(self.calado)))
                if meia_boca_atual > meia_boca_max:
                    meia_boca_max = meia_boca_atual
            self.resultados['BWL'] = meia_boca_max * 2

        # --- Cálculo do Comprimento na Linha d'Água (LWL) ---
        x_re_calc, x_vante_calc = self.casco.x_min, self.casco.x_max

        # Equação a resolver: perfil(x) - calado = 0
        funcao_raiz = lambda x: self.casco.funcao_perfil(x) - self.calado
        x_lim_re = self.casco.funcao_perfil.x.min()
        x_lim_vante = self.casco.funcao_perfil.x.max()

        # Chutes iniciais para o solver
        x_min_chute = self.casco.x_min + 1e-6
        x_max_chute = self.casco.x_max - 1e-6

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            try:
                x_re = fsolve(funcao_raiz, x0=x_min_chute)[0]
                if x_lim_re <= x_re <= x_lim_vante: x_re_calc = x_re
            except Exception:
                pass # Mantém o valor padrão se fsolve falhar

            try:
                x_vante = fsolve(funcao_raiz, x0=x_max_chute)[0]
                if x_lim_re <= x_vante <= x_lim_vante: x_vante_calc = x_vante
            except Exception:
                pass # Mantém o valor padrão se fsolve falhar
        
        self.resultados['Lim_Inf_Flut'] = min(x_re_calc, x_vante_calc)
        self.resultados['Lim_Sup_Flut'] = max(x_re_calc, x_vante_calc)
        self.resultados['LWL'] = self.resultados['Lim_Sup_Flut'] - self.resultados['Lim_Inf_Flut']

    def _calcular_area_secao(self, x: float) -> float:
        """
        Calcula a área submersa de uma seção transversal em 'x'.
        """
        # Integra 2 * meia_boca(z) de z=z_min até o calado
        def area_para_um_x(x_val: float) -> float:
            funcao_baliza = self.casco.funcoes_baliza.get(x_val)
            if not funcao_baliza:
                return 0.0
            
            # Limite inferior agora é o 'z' mínimo da baliza
            limite_inferior = self.casco.z_min_balizas.get(x_val, 0.0)

            area = integrar(
                lambda z: 2 * self.casco.obter_meia_boca(x_val, z),
                limite_inferior, self.calado
            )
            return area
        
        if isinstance(x, np.ndarray):
            v_area = np.vectorize(area_para_um_x, otypes=[float])
            return v_area(x)
        else:
            return area_para_um_x(x)

    def _calcular_momento_vertical_secao(self, x: float) -> float:
        """
        Calcula o momento vertical da área de uma seção transversal em 'x'.
        """
        # Integra 2 * meia_boca(z) * z de z=0 até o calado
        def momento_para_um_x(x_val: float) -> float:
            return integrar(
                lambda z: 2 * self.casco.obter_meia_boca(x_val, z) * z,
                0, self.calado
            )

        if isinstance(x, np.ndarray):
            return np.array([momento_para_um_x(val) for val in x])
        else:
            return momento_para_um_x(x)
    
    def _calcular_area_plano_flutuacao(self) -> float:
        """
        Calcula a área do plano de flutuação (Waterplane Area - AW).
        """
        l_inf, l_sup = self.resultados['Lim_Inf_Flut'], self.resultados['Lim_Sup_Flut']
        # Integra a boca (2 * meia_boca) ao longo do comprimento de flutuação
        aw = integrar(
            lambda x: 2 * self.casco.obter_meia_boca(x, self.calado),
            l_inf,
            l_sup
        )
        return aw

    def calcular_todas_propriedades(self) -> Dict[str, Any]:
        """
        Orquestra e executa todos os cálculos para o calado definido.
        """
        # 1. Armazena o calado de entrada
        self.resultados['Calado'] = self.calado
        
        # 2. Calcula os limites da linha d'água
        self._calcular_dimensoes_linha_dagua()

        # 3. Executa os cálculos principais (usando os métodos da classe base)
        #    Esses métodos irão, por sua vez, chamar as implementações
        #    de _calcular_area_secao e _calcular_momento_vertical_secao desta classe.
        self.resultados['Volume'], self.resultados['Desloc'] = self._calcular_volume_deslocamento()
        self.resultados['LCB'] = self._calcular_lcb()
        self.resultados['VCB'] = self._calcular_vcb()
        
        # 4. Calcula a área de flutuação (método desta classe)
        self.resultados['AW'] = self._calcular_area_plano_flutuacao()

        # 5. Calcula LCF e Momentos de Inércia (métodos da classe base)
        self.resultados['LCF'] = self._calcular_lcf()
        self.resultados['IT'] = self._calcular_momento_inercia_transversal()
        self.resultados['IL'] = self._calcular_momento_inercia_longitudinal()

        # 6. Calcula todas as propriedades derivadas (método da classe base)
        self._calcular_propriedades_derivadas()

        return self.resultados


if __name__ == '__main__':
    print("="*60)
    print("Laboratório de Teste de Hidrostática (ch_teste.py)")
    print("="*60)
    print("\nClasse 'PropriedadesHidrostaticasDireita' implementada com sucesso.")
    print("Abaixo, um exemplo de execução para um casco simplificado (caixa):")

    # --- Bloco de Teste ---
    # 1. Criar uma tabela de cotas de exemplo (uma caixa de 10x4x2 metros)
    dados_caixa = {
        'x': [0, 0, 10, 10], # Coordenadas X das balizas
        'y': [2, 2, 2, 2],   # Meia-boca (constante)
        'z': [0, 2, 0, 2]    # Altura (pontal)
    }
    tabela_cotas_caixa = pd.DataFrame(dados_caixa)
    
    # 2. Criar o objeto de geometria do casco
    casco_caixa = InterpoladorCasco(tabela_cotas_caixa, metodo_interp='linear')

    # 3. Definir um calado para o cálculo
    calado_de_teste = 1.0

    # 4. Instanciar a nova calculadora e executar os cálculos
    print(f"\nCalculando propriedades para um calado de {calado_de_teste:.2f} m...")
    calculadora_direita = PropriedadesHidrostaticasDireita(casco_caixa, calado=calado_de_teste)
    resultados = calculadora_direita.calcular_todas_propriedades()
    
    # 5. Exibir resultados formatados
    print("\n--- Resultados do Cálculo ---")
    for chave, valor in resultados.items():
        print(f"  {chave:<15}: {valor:.4f}")
    print("---------------------------\n")

    # Verificação teórica para a caixa 10x4x1
    # Volume = 10 * 4 * 1 = 40
    # Desloc = 40 * 1.025 = 41
    # LCB = 10 / 2 = 5
    # VCB = 1 / 2 = 0.5
    # AW = 10 * 4 = 40
    # LCF = 10 / 2 = 5
    print("Valores teóricos esperados para a caixa:")
    print("Volume=40.0, Desloc=41.0, LCB=5.0, VCB=0.5, AW=40.0, LCF=5.0")