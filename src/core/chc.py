import pandas as pd
import numpy as np
from scipy.integrate import quad
from scipy.optimize import fsolve
from scipy.interpolate import interp1d, PchipInterpolator
import concurrent.futures
import time
from typing import List, Dict, Any

class InterpoladorCasco:
    """
    Cria e armazena os interpoladores que definem a geometria do casco
    a partir da tabela de cotas processada.

    Esta classe funciona como uma "fábrica" que transforma uma nuvem de pontos (x, y, z)
    em um modelo matemático funcional do casco.
    """
    def __init__(self, tabela_cotas: pd.DataFrame, metodo_interp: str = 'linear'):
        """
        O construtor da classe. Ele recebe os dados e orquestra a criação
        dos interpoladores.

        Args:
            tabela_cotas (pd.DataFrame): O DataFrame já validado e processado
                                         pelo FileHandler.
            metodo_interp (str, optional): O método de interpolação a ser usado.
                                           Defaults to 'linear'.
        """
        # --- 1. Armazena as propriedades básicas ---
        self.metodo_interp = metodo_interp
        
        # Armazena uma lista ordenada e única das posições 'x' (balizas/estações)
        # Isso é útil para saber os limites longitudinais do casco.
        self.posicoes_balizas: List[float] = sorted(tabela_cotas['x'].unique())
        
        # --- 2. Cria os interpoladores ---
        # Cria um dicionário onde cada chave é uma posição 'x' e o valor é a
        # função que descreve a forma daquela seção transversal (y em função de z).
        self.funcoes_baliza: Dict[float, Any] = self._criar_interpoladores_balizas(tabela_cotas)
        
        # Cria uma única função que descreve o perfil longitudinal do casco
        # (a linha da quilha, z em função de x). Isso é crucial para encontrar o LWL.
        self.funcao_perfil: Any = self._criar_interpolador_perfil(tabela_cotas)

    def _criar_interpoladores_balizas(self, df: pd.DataFrame) -> Dict[float, Any]:
        """
        Método auxiliar para criar as funções de interpolação para cada baliza.
        Uma baliza (ou seção transversal) é definida pela sua forma no plano Y-Z.
        Esta função cria um interpolador para y = f(z) para cada estação x.
        """
        funcoes = {}
        # O método `groupby('x')` do pandas é perfeito para isso. Ele itera sobre o
        # DataFrame, nos dando a posição 'x' e um sub-DataFrame ('grupo') contendo
        # todos os pontos (y, z) apenas para aquela estação.
        for x_pos, grupo in df.groupby('x'):
            
            # --- Bloco de segurança para garantir a forma correta do casco na quilha ---
            # Se uma baliza não tem um ponto definido em z=0 (quilha), a interpolação
            # poderia criar uma forma irreal perto do fundo. Forçamos a adição de um
            # ponto (y=0, z=0) para garantir que o casco feche corretamente na quilha.
            if 0 not in grupo['z'].values:
                ponto_quilha = pd.DataFrame([{'x': x_pos, 'y': 0, 'z': 0}])
                grupo = pd.concat([ponto_quilha, grupo], ignore_index=True)
            
            # Ordena os pontos pela altura (z) e remove duplicatas, o que é um
            # requisito para a função 'interp1d'.
            grupo = grupo.sort_values(by='z').drop_duplicates(subset='z')
            
            # Precisamos de pelo menos 2 pontos para criar uma linha (e uma interpolação).
            if len(grupo) >= 2:
                # Esta é a linha principal! `interp1d` pega os pontos (z, y) e
                # retorna um objeto que se comporta como uma função.
                # Ex: `minha_funcao = interp1d([0, 1, 2], [0, 0.5, 0.8])`
                #     `minha_funcao(1.5)` -> retornaria um y interpolado para z=1.5
                funcoes[x_pos] = interp1d(
                    grupo['z'],          # O eixo independente (altura)
                    grupo['y'],          # O eixo dependente (meia-boca)
                    kind=self.metodo_interp,
                    bounds_error=False,  # Não gera erro se pedirmos um ponto fora do intervalo
                    fill_value=0.0       # Retorna 0.0 para pontos fora do intervalo (ex: acima do convés)
                )
        return funcoes

    def _criar_interpolador_perfil(self, df: pd.DataFrame) -> Any:
        """
        Método auxiliar para criar a função de interpolação para o perfil do navio.
        O perfil é a linha formada pelos pontos mais altos de cada baliza.
        Isso nos dá uma função z = f(x) para a linha do convés.
        """
        # Esta é uma forma eficiente de obter o ponto mais alto (máximo 'z') de cada baliza.
        # 1. `groupby('x')`: Agrupa os dados por estação.
        # 2. `['z'].idxmax()`: Para cada grupo, encontra o índice da linha com o maior 'z'.
        # 3. `df.loc[...]`: Seleciona todas essas linhas do DataFrame original.
        perfil_df = df.loc[df.groupby('x')['z'].idxmax()]
        
        if len(perfil_df) < 2:
            return None # Não é possível interpolar com menos de 2 pontos.
            
        # Criamos a interpolação do perfil, que nos dá a altura do convés (z)
        # para qualquer ponto ao longo do comprimento (x).
        return interp1d(
            perfil_df['x'], 
            perfil_df['z'], 
            kind=self.metodo_interp, 
            bounds_error=False, 
            fill_value=0.0
        )

class PropriedadesHidrostaticas:
    """
    Calcula e armazena todas as propriedades hidrostáticas para um único calado.
    """
    def __init__(self, interpolador_casco: InterpoladorCasco, calado: float, densidade: float):
        self.casco = interpolador_casco
        self.calado = calado
        self.densidade = densidade
        self.metodo_interp = self.casco.metodo_interp # Pega da instância do casco
        
        self.areas_secoes = {}
        # ... (todos os outros atributos de PropriedadesHidrostaticas, como lwl, bwl, volume, etc.)
        self.lwl = self.bwl = self.area_plano_flutuacao = self.volume = self.deslocamento = 0.0
        self.lcf = self.lcb = self.vcb = self.bmt = self.kmt = self.bml = self.kml = 0.0
        self.tpc = self.mtc = self.cb = self.cp = self.cwp = self.cm = 0.0
        
        self._calcular_todas_propriedades()

    def obter_meia_boca(self, x: float, z: float) -> float:
        """
        Interpola a meia-boca (y) para uma dada posição longitudinal (x) e vertical (z).
        """
        if z > self.calado or z < 0:
            return 0.0

        x_balizas = self.casco.posicoes_balizas
        if x < x_balizas[0] or x > x_balizas[-1]:
            return 0.0

        # Encontra as balizas adjacentes
        idx = np.searchsorted(x_balizas, x)
        if idx == 0 or x == x_balizas[idx]:
            x1 = x_balizas[idx]
            funcao1 = self.casco.funcoes_baliza.get(x1)
            return np.nan_to_num(float(funcao1(z))) if funcao1 else 0.0

        x1, x2 = x_balizas[idx - 1], x_balizas[idx]
        funcao1 = self.casco.funcoes_baliza.get(x1)
        funcao2 = self.casco.funcoes_baliza.get(x2)

        y1 = np.nan_to_num(float(funcao1(z))) if funcao1 else 0.0
        y2 = np.nan_to_num(float(funcao2(z))) if funcao2 else 0.0

        # Interpolação linear entre as balizas
        return y1 + (y2 - y1) * (x - x1) / (x2 - x1)
    
    # --- Colar aqui todos os métodos da classe PropriedadesHidrostaticas ---
    # _calcular_dimensoes_linha_dagua
    # _calcular_area_secao
    # _calcular_area_plano_flutuacao
    # _calcular_volume_deslocamento
    # _calcular_lcf
    # _calcular_lcb
    # _calcular_momento_vertical_secao
    # _calcular_vcb
    # _calcular_momento_inercia_transversal
    # _calcular_momento_inercia_longitudinal
    # _calcular_propriedades_derivadas
    # _calcular_todas_propriedades
    # (Copie-os exatamente como estão no arquivo original, 
    #  mas mude `self.casco.obter_meia_boca` para `self.obter_meia_boca` dentro deles)
    
    # Exemplo da mudança necessária:
    def _calcular_area_secao(self, x_baliza: float) -> float:
        area, erro = quad(lambda z: self.obter_meia_boca(x_baliza, z), 0, self.calado)
        return area * 2
    
    # ... (continuação dos métodos copiados e adaptados)
    def _calcular_dimensoes_linha_dagua(self):
        meia_boca_max = 0.0
        for funcao_baliza in self.casco.funcoes_baliza.values():
            meia_boca_atual = np.nan_to_num(float(funcao_baliza(self.calado)))
            if meia_boca_atual > meia_boca_max:
                meia_boca_max = meia_boca_atual
        self.bwl = meia_boca_max * 2

        if self.casco.funcao_perfil:
            funcao_raiz = lambda x: self.casco.funcao_perfil(x) - self.calado
            x_lim_re = self.casco.funcao_perfil.x.min()
            x_lim_vante = self.casco.funcao_perfil.x.max()
            try:
                x_re_calc = fsolve(funcao_raiz, x0=x_lim_re + 1e-6)[0]
                if not (x_lim_re <= x_re_calc <= x_lim_vante): x_re_calc = x_lim_re
            except: x_re_calc = x_lim_re
            try:
                x_vante_calc = fsolve(funcao_raiz, x0=x_lim_vante - 1e-6)[0]
                if not (x_lim_re <= x_vante_calc <= x_lim_vante): x_vante_calc = x_lim_vante
            except: x_vante_calc = x_lim_vante
            self.x_re, self.x_vante = min(x_re_calc, x_vante_calc), max(x_re_calc, x_vante_calc)
            self.lwl = self.x_vante - self.x_re
        else: self.lwl, self.x_re, self.x_vante = 0.0, 0.0, 0.0

    def _calcular_area_plano_flutuacao(self):
        balizas_internas_x = [x for x in self.casco.posicoes_balizas if x > self.x_re and x < self.x_vante]
        balizas_internas_y = [self.obter_meia_boca(x, self.calado) for x in balizas_internas_x]
        baliza_popa_x, baliza_proa_x, tolerancia = min(self.casco.posicoes_balizas), max(self.casco.posicoes_balizas), 1e-3
        y_re = self.obter_meia_boca(baliza_popa_x, self.calado) if abs(self.x_re - baliza_popa_x) < tolerancia else 0.0
        y_vante = self.obter_meia_boca(baliza_proa_x, self.calado) if abs(self.x_vante - baliza_proa_x) < tolerancia else 0.0
        x_pontos, y_pontos = [self.x_re] + balizas_internas_x + [self.x_vante], [y_re] + balizas_internas_y + [y_vante]
        if len(x_pontos) < 2: self.area_plano_flutuacao, self.interpolador_wl = 0.0, None; return
        pontos_unicos = sorted(list(set(zip(x_pontos, y_pontos))))
        x_pontos_unicos, y_pontos_unicos = [p[0] for p in pontos_unicos], [p[1] for p in pontos_unicos]
        self.interpolador_wl = PchipInterpolator(x_pontos_unicos, y_pontos_unicos, extrapolate=False) if self.metodo_interp == 'pchip' else interp1d(x_pontos_unicos, y_pontos_unicos, kind='linear', bounds_error=False, fill_value=0.0)
        meia_area, _ = quad(self.interpolador_wl, self.x_re, self.x_vante)
        self.area_plano_flutuacao = meia_area * 2

    def _calcular_volume_deslocamento(self):
        x_internos = [x for x in self.casco.posicoes_balizas if x > self.x_re and x < self.x_vante]
        areas_internas = [self.areas_secoes[x] for x in x_internos]
        baliza_popa_x, baliza_proa_x, tolerancia = min(self.casco.posicoes_balizas), max(self.casco.posicoes_balizas), 1e-3
        area_re = self.areas_secoes.get(baliza_popa_x, 0.0) if abs(self.x_re - baliza_popa_x) < tolerancia else 0.0
        area_vante = self.areas_secoes.get(baliza_proa_x, 0.0) if abs(self.x_vante - baliza_proa_x) < tolerancia else 0.0
        x_pontos, areas_pontos = [self.x_re] + x_internos + [self.x_vante], [area_re] + areas_internas + [area_vante]
        if len(x_pontos) < 2: self.volume, self.deslocamento = 0.0, 0.0; return
        pontos_unicos = sorted(list(set(zip(x_pontos, areas_pontos))))
        x_pontos_unicos, areas_pontos_unicos = [p[0] for p in pontos_unicos], [p[1] for p in pontos_unicos]
        self.interpolador_areas = PchipInterpolator(x_pontos_unicos, areas_pontos_unicos, extrapolate=False) if self.metodo_interp == 'pchip' else interp1d(x_pontos_unicos, areas_pontos_unicos, kind='linear', bounds_error=False, fill_value=0.0)
        volume_calculado, _ = quad(self.interpolador_areas, self.x_re, self.x_vante)
        self.volume, self.deslocamento = volume_calculado, volume_calculado * self.densidade

    def _calcular_lcf(self):
        if self.area_plano_flutuacao == 0.0 or not hasattr(self, 'interpolador_wl'): self.lcf = 0.0; return
        momento_long_meia_area, _ = quad(lambda x: x * self.interpolador_wl(x), self.x_re, self.x_vante)
        meia_area = self.area_plano_flutuacao / 2
        self.lcf = momento_long_meia_area / meia_area if meia_area > 1e-6 else 0.0

    def _calcular_lcb(self):
        if self.volume == 0.0 or not hasattr(self, 'interpolador_areas'): self.lcb = 0.0; return
        momento_long_volume, _ = quad(lambda x: x * self.interpolador_areas(x), self.x_re, self.x_vante)
        self.lcb = momento_long_volume / self.volume if abs(self.volume) > 1e-6 else 0.0

    def _calcular_momento_vertical_secao(self, x_baliza: float) -> float:
        momento_vertical, _ = quad(lambda z: z * 2 * self.obter_meia_boca(x_baliza, z), 0, self.calado)
        return momento_vertical

    def _calcular_vcb(self):
        if self.volume == 0.0: self.vcb = 0.0; return
        momentos_verticais = {x: self._calcular_momento_vertical_secao(x) for x in self.casco.posicoes_balizas}
        x_pontos_sorted, momentos_pontos_sorted = zip(*sorted(momentos_verticais.items()))
        interpolador_momentos = PchipInterpolator(x_pontos_sorted, momentos_pontos_sorted, extrapolate=False) if self.metodo_interp == 'pchip' else interp1d(x_pontos_sorted, momentos_pontos_sorted, kind='linear', bounds_error=False, fill_value=0.0)
        momento_total_vertical, _ = quad(interpolador_momentos, self.x_re, self.x_vante)
        self.vcb = momento_total_vertical / self.volume if abs(self.volume) > 1e-6 else 0.0
    
    def _calcular_momento_inercia_transversal(self):
        x_internos = [x for x in self.casco.posicoes_balizas if x > self.x_re and x < self.x_vante]
        y_cubed_internas = [self.obter_meia_boca(x, self.calado)**3 for x in x_internos]
        baliza_popa_x, baliza_proa_x, tolerancia = min(self.casco.posicoes_balizas), max(self.casco.posicoes_balizas), 1e-3
        y_cubed_re = self.obter_meia_boca(baliza_popa_x, self.calado)**3 if abs(self.x_re - baliza_popa_x) < tolerancia else 0.0
        y_cubed_vante = self.obter_meia_boca(baliza_proa_x, self.calado)**3 if abs(self.x_vante - baliza_proa_x) < tolerancia else 0.0
        x_pontos, y_cubed_pontos = [self.x_re] + x_internos + [self.x_vante], [y_cubed_re] + y_cubed_internas + [y_cubed_vante]
        if len(x_pontos) < 2: self.momento_inercia_transversal = 0.0; return
        pontos_unicos = sorted(list(set(zip(x_pontos, y_cubed_pontos))))
        x_pontos_unicos, y_cubed_pontos_unicos = [p[0] for p in pontos_unicos], [p[1] for p in pontos_unicos]
        interpolador_y3 = PchipInterpolator(x_pontos_unicos, y_cubed_pontos_unicos, extrapolate=False) if self.metodo_interp == 'pchip' else interp1d(x_pontos_unicos, y_cubed_pontos_unicos, kind='linear', bounds_error=False, fill_value=0.0)
        integral_y3, _ = quad(interpolador_y3, self.x_re, self.x_vante)
        self.momento_inercia_transversal = (2/3) * integral_y3

    def _calcular_momento_inercia_longitudinal(self):
        if self.area_plano_flutuacao == 0.0 or not hasattr(self, 'interpolador_wl') or self.lcf is None: self.momento_inercia_longitudinal = 0.0; return
        momento_meia_area, _ = quad(lambda x: ((x - self.lcf)**2) * self.interpolador_wl(x), self.x_re, self.x_vante)
        self.momento_inercia_longitudinal = momento_meia_area * 2

    def _calcular_propriedades_derivadas(self):
        self.bmt = self.momento_inercia_transversal / self.volume if self.volume > 1e-6 else 0.0
        self.kmt = self.vcb + self.bmt
        self.bml = self.momento_inercia_longitudinal / self.volume if self.volume > 1e-6 else 0.0
        self.kml = self.vcb + self.bml
        self.tpc = (self.area_plano_flutuacao * self.densidade) / 100.0
        self.mtc = (self.momento_inercia_longitudinal * self.densidade) / (100 * self.lwl) if self.lwl > 1e-6 else 0.0
        denominador_bloco = self.lwl * self.bwl * self.calado
        self.cb = self.volume / denominador_bloco if denominador_bloco > 1e-6 else 0.0
        area_secao_mestra = max(self.areas_secoes.values()) if self.areas_secoes else 0.0
        denominador_prismatico = area_secao_mestra * self.lwl
        self.cp = self.volume / denominador_prismatico if denominador_prismatico > 1e-6 else 0.0
        denominador_plano_flutuacao = self.lwl * self.bwl
        self.cwp = self.area_plano_flutuacao / denominador_plano_flutuacao if denominador_plano_flutuacao > 1e-6 else 0.0
        self.cm = self.cb / self.cp if self.cp > 1e-6 else 0.0

    def _calcular_todas_propriedades(self):
        print(f"Calculando propriedades para o calado T = {self.calado:.3f} m...")
        self._calcular_dimensoes_linha_dagua()
        self._calcular_area_plano_flutuacao()
        self._calcular_lcf()
        for x_pos in self.casco.posicoes_balizas:
            self.areas_secoes[x_pos] = self._calcular_area_secao(x_pos)
        self._calcular_volume_deslocamento()
        self._calcular_lcb()
        self._calcular_vcb()
        self._calcular_momento_inercia_transversal()
        self._calcular_momento_inercia_longitudinal()
        self._calcular_propriedades_derivadas()

def calcular_propriedades_para_um_calado(args):
    """Função 'worker' para o multiprocessing."""
    interpolador, calado, densidade = args
    props = PropriedadesHidrostaticas(interpolador, calado, densidade)
    return {
        'Calado (m)': calado, 'Volume (m³)': props.volume, 'Desloc. (t)': props.deslocamento,
        'AWP (m²)': props.area_plano_flutuacao, 'LWL (m)': props.lwl, 'BWL (m)': props.bwl,
        'LCB (m)': props.lcb, 'VCB (m)': props.vcb, 'LCF (m)': props.lcf,
        'BMt (m)': props.bmt, 'KMt (m)': props.kmt, 'BMl (m)': props.bml, 'KMl (m)': props.kml,
        'TPC (t/cm)': props.tpc, 'MTc (t·m/cm)': props.mtc, 'Cb': props.cb, 'Cp': props.cp,
        'Cwp': props.cwp, 'Cm': props.cm,
    }

class CalculadoraHidrostatica:
    """Orquestra o cálculo das curvas hidrostáticas para múltiplos calados."""
    def __init__(self, interpolador_casco: InterpoladorCasco, densidade: float):
        self.casco = interpolador_casco
        self.densidade = densidade
        
    def calcular_curvas(self, lista_de_calados: list) -> pd.DataFrame:
        start_time = time.perf_counter()
        print(f"\nIniciando cálculo PARALELO para {len(lista_de_calados)} calados...")
        
        tarefas = [(self.casco, calado, self.densidade) for calado in lista_de_calados]
        
        with concurrent.futures.ProcessPoolExecutor() as executor:
            resultados = list(executor.map(calcular_propriedades_para_um_calado, tarefas))
            
        duration = time.perf_counter() - start_time
        print(f"Cálculo finalizado em {duration:.2f} segundos.")
        
        resultados.sort(key=lambda r: r['Calado (m)'])
        return pd.DataFrame(resultados)