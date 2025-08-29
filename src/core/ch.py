import pandas as pd
import numpy as np
from scipy.interpolate import interp1d, PchipInterpolator
from scipy.optimize import fsolve
from src.utils.integration import integrar
from scipy.integrate import quad
from typing import Dict, Any, List
import concurrent.futures
import time

class InterpoladorCasco:
    """
    Representa a geometria do casco através de funções de interpolação.

    Esta classe encapsula a lógica de transformação de uma tabela de cotas
    discreta em um modelo geométrico contínuo. Ela serve como a definição
    matemática do casco, essencial para as integrações numéricas nos cálculos
    hidrostáticos.

    Métodos de Interpolação:
    - Linear: Conecta os pontos de dados com segmentos de reta. É um método
      robusto e computacionalmente eficiente.
    - PCHIP (Piecewise Cubic Hermite Interpolating Polynomial): Utiliza splines
      cúbicas que preservam a monotonia dos dados, resultando em uma curva suave
      que não apresenta oscilações entre os pontos.

    Referência: A representação matemática de formas de casco a partir de dados
    discretos é um conceito fundamental em arquitetura naval. Para mais detalhes,
    consulte: Principles of Naval Architecture (Second Revision), Volume I,
    Chapter I, Section 2.
    """
    def __init__(self, tabela_cotas: pd.DataFrame, metodo_interp: str = 'linear'):
        """
        Inicializa o objeto InterpoladorCasco.

        Args:
            tabela_cotas (pd.DataFrame): DataFrame com as coordenadas (x, y, z)
                                         do casco, já processado e validado.
            metodo_interp (str, optional): Método de interpolação: 'linear' ou 'pchip'.
        """
        self.tabela_cotas: pd.DataFrame = tabela_cotas
        self.metodo_interp: str = metodo_interp
        self.posicoes_balizas: List[float] = sorted(self.tabela_cotas['x'].unique())
        self.funcoes_baliza: Dict[float, Any] = self._gerar_interpoladores_secao()
        self.funcao_perfil: Any = self._gerar_interpolador_perfil()

    def _gerar_interpoladores_secao(self) -> Dict[float, Any]:
        """
        Gera um dicionário de funções para cada seção transversal (y=f(z))
        iterando sobre cada posição longitudinal.
        """
        interpoladores = {}
        # Itera sobre a lista de posições 'x' únicas, como no código de referência.
        for x_val in self.posicoes_balizas:
            # Filtra o DataFrame principal para obter os pontos de uma única baliza
            # e os ordena pela coordenada vertical 'z'.
            df_baliza = self.tabela_cotas[self.tabela_cotas['x'] == x_val].sort_values('z')

            z_coords = df_baliza['z'].values
            y_coords = df_baliza['y'].values

            # A interpolação é viável apenas com um mínimo de dois pontos.
            if len(z_coords) > 1:
                if self.metodo_interp == 'pchip':
                    # PCHIP é chamado como uma classe para criar o interpolador.
                    interpolador = PchipInterpolator(z_coords, y_coords, extrapolate=False)
                else:
                    # interp1d é a função para interpolação linear.
                    interpolador = interp1d(z_coords, y_coords, kind='linear', bounds_error=False, fill_value=0.0)

                # Armazena a função gerada no dicionário.
                interpoladores[x_val] = interpolador
        
        # Retorna o dicionário completo de funções de interpolação.
        return interpoladores

    def _gerar_interpolador_perfil(self) -> Any:
        """
        Gera a função de interpolação para o perfil longitudinal da quilha (z=f(x)).

        Este método define a linha de fundo do casco utilizando os pontos de menor
        cota vertical de cada estação para criar um perfil contínuo.
        """
        # Agrupa o DataFrame por estação 'x' e agrega encontrando o valor mínimo de 'z'
        # para cada uma, extraindo eficientemente os pontos da linha da quilha.
        dados_perfil = self.tabela_cotas.groupby('x', as_index=False).agg(z_min=('z', 'min'))
        dados_perfil = dados_perfil.sort_values('x')

        # A interpolação é viável apenas com um mínimo de dois pontos.
        if len(dados_perfil['x']) > 1:
            x_coords = dados_perfil['x'].values
            z_coords = dados_perfil['z_min'].values

            if self.metodo_interp == 'pchip':
                # PCHIP é chamado como uma classe para criar o interpolador.
                return PchipInterpolator(x_coords, z_coords, extrapolate=False)
            else:
                # interp1d é a função para interpolação linear.
                return interp1d(x_coords, z_coords, kind='linear', bounds_error=False, fill_value=0.0)
            
        # Retorna None se não for possível gerar um interpolador.
        return None    
        
class PropriedadesHidrostaticas:
    """
    Calcula e armazena todas as propriedades hidrostáticas para um único calado.

    Esta classe utiliza um objeto InterpoladorCasco para acessar a geometria
    do navio e calcular suas características de flutuação, estabilidade e forma
    para uma condição de carregamento específica (definida pelo calado).
    """
    def __init__(self, interpolador_casco: InterpoladorCasco, calado: float, densidade: float):
        """
        Inicializa o objeto de cálculo para um calado específico.

        Args:
            interpolador_casco (InterpoladorCasco): O objeto que contém a geometria
                                                    funcional do casco.
            calado (float): O calado (T) para o qual as propriedades serão calculadas [m].
            densidade (float): A densidade da água [t/m³].
        """
        self.casco: InterpoladorCasco = interpolador_casco
        self.calado: float = calado
        self.densidade: float = densidade
        
        # Atributos que serão calculados pelos métodos
        self.lwl: float = 0.0  # Comprimento na linha d'água
        self.bwl: float = 0.0  # Boca na linha d'água
        self.x_re: float = 0.0   # Limite longitudinal de ré da linha d'água
        self.x_vante: float = 0.0 # Limite longitudinal de vante da linha d'água
        self.areas_secoes: Dict[float, float] = {} # Dicionário para armazenar as áreas de cada seção transversal calculada
        self.area_plano_flutuacao: float = 0.0 # Atributo para a área do plano de flutuação (Waterplane Area)
        self.volume: float = 0.0 # Atributo para o volume de deslocamento
        self.deslocamento: float = 0.0 # Atributo para o deslocamento em massa
        self.interpolador_wl: Any = None # Atributo para o interpolador da linha d'água
        self.interpolador_areas: Any = None # Atributo para o interpolador das áreas seccionais
        self.lcf: float = 0.0  # Posição Longitudinal do Centro de Flutuação
        self.lcb: float = 0.0  # Posição Longitudinal do Centro de Carena
        self.vcb: float = 0.0  # Posição Vertical do Centro de Carena
        self.momento_inercia_transversal: float = 0.0
        self.momento_inercia_longitudinal: float = 0.0
        self.bmt: float = 0.0  # Raio Metacêntrico Transversal
        self.kmt: float = 0.0  # Altura Metacêntrica Transversal
        self.bml: float = 0.0  # Raio Metacêntrico Longitudinal
        self.kml: float = 0.0  # Altura Metacêntrica Longitudinal
        self.tpc: float = 0.0  # Toneladas por Centímetro de Imersão
        self.mtc: float = 0.0  # Momento para Alterar o Trim em 1 cm
        self.cb: float = 0.0   # Coeficiente de Bloco
        self.cp: float = 0.0   # Coeficiente Prismático
        self.cwp: float = 0.0  # Coeficiente do Plano de Flutuação
        self.cm: float = 0.0   # Coeficiente de Seção Mestra

        self._calcular_todas_propriedades()

    def _obter_meia_boca(self, x: float, z: float) -> float:
        """
        Obtém a meia-boca (y) para uma dada estação 'x' e altura 'z' existentes.

        Este método consulta diretamente o dicionário de interpoladores para uma estação
        específica.

        Args:
            x (float): A coordenada longitudinal da estação a ser consultada.
            z (float): A coordenada vertical na qual a meia-boca é desejada.

        Returns:
            float: O valor da meia-boca (y). Retorna 0 se a estação 'x' não
                possuir um interpolador definido no dicionário.
        """
        # Recupera a função de interpolação pré-calculada para a estação 'x'.
        # O método .get() retorna None se a chave 'x' não for encontrada.
        funcao_interpoladora = self.casco.funcoes_baliza.get(x)
        
        # Verifica se um interpolador foi encontrado para a estação solicitada.
        if funcao_interpoladora:
            # Invoca a função de interpolação para calcular a meia-boca na altura 'z'.
            meia_boca = funcao_interpoladora(z)
            
            # Converte um possível resultado NaN para 0.0, garantindo um retorno numérico.
            # Isso é importante para interpoladores como PCHIP com extrapolate=False.
            return np.nan_to_num(meia_boca)
        else:
            # Se a baliza não existe, retorna um valor compatível.
            # Se 'z' for um array, retorna um array de zeros com o mesmo formato.
            if isinstance(z, np.ndarray):
                return np.zeros_like(z)
            return 0.0
        
    def _calcular_dimensoes_linha_dagua(self):
        """
        Calcula as dimensões principais da linha d'água (LWL e BWL).

        - BWL (Boca na Linha d'Água): É determinada iterando-se sobre todas as
          seções transversais e encontrando a máxima meia-boca na altura do calado atual.
        - LWL (Comprimento na Linha d'Água): É encontrado numericamente,
          buscando as interseções (raízes) entre a função do perfil do casco
          e a linha d'água (z = calado).
        """
        # --- Cálculo da Boca na Linha d'Água (BWL) ---
        meia_boca_max = 0.0
        # Itera sobre todas as funções de interpolação de baliza disponíveis.
        for funcao_baliza in self.casco.funcoes_baliza.values():
            # Para cada baliza, calcula a meia-boca na altura exata do calado.
            meia_boca_atual = np.nan_to_num(float(funcao_baliza(self.calado)))
            # Atualiza o valor máximo encontrado.
            if meia_boca_atual > meia_boca_max:
                meia_boca_max = meia_boca_atual
        # A boca total (BWL) é o dobro da máxima meia-boca.
        self.bwl = meia_boca_max * 2

        # --- Cálculo do Comprimento na Linha d'Água (LWL) ---
        # Verifica se o interpolador do perfil do casco foi criado com sucesso.
        if self.casco.funcao_perfil:
            # Define uma função cujo zero corresponde à interseção do perfil com o calado.
            # f(x) = z_perfil(x) - calado. Queremos encontrar x tal que f(x) = 0.
            funcao_raiz = lambda x: self.casco.funcao_perfil(x) - self.calado
            
            # Obtém os limites longitudinais do casco para guiar a busca pelas raízes.
            x_lim_re = self.casco.funcao_perfil.x.min()
            x_lim_vante = self.casco.funcao_perfil.x.max()

            # Utiliza 'fsolve' da biblioteca SciPy, um solver numérico, para encontrar as raízes.
            try:
                # Procura a interseção de ré, iniciando a busca próximo ao limite de ré.
                x_re_calc = fsolve(funcao_raiz, x0=x_lim_re + 1e-6)[0]
                # Valida se a raiz encontrada está dentro dos limites do navio.
                if not (x_lim_re <= x_re_calc <= x_lim_vante): x_re_calc = x_lim_re
            except:
                # Se o solver falhar, assume o limite do casco como a interseção.
                x_re_calc = x_lim_re

            try:
                # Procura a interseção de vante, iniciando a busca próximo ao limite de vante.
                x_vante_calc = fsolve(funcao_raiz, x0=x_lim_vante - 1e-6)[0]
                if not (x_lim_re <= x_vante_calc <= x_lim_vante): x_vante_calc = x_lim_vante
            except:
                x_vante_calc = x_lim_vante

            # Armazena os limites da linha d'água.
            self.x_re = min(x_re_calc, x_vante_calc)
            self.x_vante = max(x_re_calc, x_vante_calc)
            # O LWL é a distância entre os pontos de interseção.
            self.lwl = self.x_vante - self.x_re

    def _calcular_area_secao(self, x_baliza: float) -> float:
        """
        Calcula a área submersa de uma única seção transversal (baliza).

        A área é obtida pela integração numérica da função da meia-boca (y(z))
        desde a quilha (z=0) até o calado atual. A função `quad` da biblioteca
        SciPy é utilizada para realizar a integração. O resultado é multiplicado
        por dois para obter a área total da seção (bombordo e estibordo).

        Referência: O cálculo de áreas seccionais por integração é um procedimento
        padrão. Ver "Principles of Naval Architecture", Vol. I, Cap. I, Seção 3.

        Args:
            x_baliza (float): A posição longitudinal (X) da baliza.

        Returns:
            float: A área da seção transversal submersa [m²].
        """
        # A integral de 2 * y(z) dz de 0 a T, pode ser calculada como
        # 2 * integral de y(z) dz. A função a ser integrada é a meia-boca.
        funcao_a_integrar = lambda z: self._obter_meia_boca(x_baliza, z)
        area_meio_casco = integrar(funcao_a_integrar, 0, self.calado)
        
        # A área total da seção é o dobro da área de meio casco.
        return area_meio_casco * 2
    
    def _calcular_area_plano_flutuacao(self):
        """
        Calcula a área do plano de flutuação (AWP).

        Este método primeiro constrói um interpolador que descreve a forma da
        linha d'água (curva de meia-boca y(x) no calado atual). Em seguida,
        integra numericamente esta função ao longo do LWL para obter a área.
        Lógica especial é aplicada para tratar as extremidades do casco.
        """
        # 1. Obtém as meias-bocas no calado atual para todas as balizas internas
        #    (que estão estritamente dentro dos limites da linha d'água).
        balizas_internas_x = [
            x for x in self.casco.posicoes_balizas 
            if x > self.x_re and x < self.x_vante
        ]
        balizas_internas_y = [self._obter_meia_boca(x, self.calado) for x in balizas_internas_x]

        # 2. Determina as meias-bocas nas extremidades (x_re e x_vante),
        #    considerando que a meia-boca pode ser zero se a linha d'água não
        #    coincidir com uma baliza física (ex: proa lançada).
        baliza_popa_x = min(self.casco.posicoes_balizas)
        baliza_proa_x = max(self.casco.posicoes_balizas)
        tolerancia = 1e-3
        y_re = self._obter_meia_boca(baliza_popa_x, self.calado) if abs(self.x_re - baliza_popa_x) < tolerancia else 0.0
        y_vante = self._obter_meia_boca(baliza_proa_x, self.calado) if abs(self.x_vante - baliza_proa_x) < tolerancia else 0.0

        # 3. Constrói as listas de pontos (x, y) que definem a linha d'água.
        x_pontos = [self.x_re] + balizas_internas_x + [self.x_vante]
        y_pontos = [y_re] + balizas_internas_y + [y_vante]
        
        if len(x_pontos) < 2: return # Impossível interpolar.

        # Garante pontos únicos e ordenados para o interpolador.
        pontos_unicos = sorted(list(set(zip(x_pontos, y_pontos))))
        x_pontos_unicos, y_pontos_unicos = [p[0] for p in pontos_unicos], [p[1] for p in pontos_unicos]

        # 4. Cria o interpolador para a linha d'água (y=f(x)).
        if self.casco.metodo_interp == 'pchip':
            self.interpolador_wl = PchipInterpolator(x_pontos_unicos, y_pontos_unicos, extrapolate=False)
        else:
            self.interpolador_wl = interp1d(x_pontos_unicos, y_pontos_unicos, kind='linear', bounds_error=False, fill_value=0.0)

        # 5. Integra o interpolador da linha d'água para obter a meia-área.
        meia_area = integrar(self.interpolador_wl, self.x_re, self.x_vante)
        self.area_plano_flutuacao = meia_area * 2
        

    def _calcular_volume_deslocamento(self):
        """
        Calcula o volume submerso e o deslocamento.

        O método se baseia no princípio de Bonjean, integrando a curva de áreas
        das seções transversais (A(x)) ao longo do comprimento da linha d'água (LWL).
        Primeiro, um interpolador para A(x) é criado, e então este é integrado
        numericamente para encontrar o volume total do casco submerso.
        """
        # 1. Constrói um interpolador para a curva de áreas seccionais (A=f(x)).
        #    As áreas nas extremidades são tratadas de forma análoga ao AWP.
        x_internos = [x for x in self.casco.posicoes_balizas if x > self.x_re and x < self.x_vante]
        areas_internas = [self.areas_secoes[x] for x in x_internos]
        baliza_popa_x, baliza_proa_x = min(self.casco.posicoes_balizas), max(self.casco.posicoes_balizas)
        tolerancia = 1e-3
        area_re = self.areas_secoes.get(baliza_popa_x, 0.0) if abs(self.x_re - baliza_popa_x) < tolerancia else 0.0
        area_vante = self.areas_secoes.get(baliza_proa_x, 0.0) if abs(self.x_vante - baliza_proa_x) < tolerancia else 0.0
        x_pontos = [self.x_re] + x_internos + [self.x_vante]
        areas_pontos = [area_re] + areas_internas + [area_vante]

        if len(x_pontos) < 2: return

        pontos_unicos = sorted(list(set(zip(x_pontos, areas_pontos))))
        x_pontos_unicos, areas_pontos_unicos = [p[0] for p in pontos_unicos], [p[1] for p in pontos_unicos]

        # 2. Cria o interpolador A(x).
        if self.casco.metodo_interp == 'pchip':
            self.interpolador_areas = PchipInterpolator(x_pontos_unicos, areas_pontos_unicos, extrapolate=False)
        else:
            self.interpolador_areas = interp1d(x_pontos_unicos, areas_pontos_unicos, kind='linear', bounds_error=False, fill_value=0.0)
        
        # 3. Integra a curva de áreas seccionais para obter o volume.
        volume_calculado = integrar(self.interpolador_areas, self.x_re, self.x_vante)
        self.volume = volume_calculado
        
        # 4. Calcula o deslocamento (massa) a partir do volume e da densidade.
        self.deslocamento = self.volume * self.densidade

    def _calcular_lcf(self):
        """
        Calcula a posição longitudinal do centro de flutuação (LCF).

        O LCF é o centroide da área do plano de flutuação (AWP). É calculado
        integrando-se o primeiro momento de área da AWP em relação à origem (x=0)
        e dividindo-se pela área total da AWP. A integral é de x * 2y(x) dx,
        onde y(x) é a curva de meia-boca da linha d'água.
        """
        # A validação previne divisão por zero ou cálculos sem um interpolador definido.
        if self.area_plano_flutuacao < 1e-6 or not self.interpolador_wl:
            self.lcf = 0.0
            return
            
        # Define a função a ser integrada: x * largura_total(x) = x * 2y(x)
        funcao_momento_longitudinal = lambda x: x * (2 * self.interpolador_wl(x))
        
        # Integra para obter o momento longitudinal total da AWP.
        momento_long_total = integrar(funcao_momento_longitudinal, self.x_re, self.x_vante)
        
        # LCF é o momento dividido pela área.
        self.lcf = momento_long_total / self.area_plano_flutuacao

    def _calcular_lcb(self):
        """
        Calcula a posição longitudinal do centro de carena (LCB).

        O LCB é o centroide do volume submerso. De forma análoga ao LCF, ele é
        calculado integrando-se o primeiro momento de volume em relação à
        origem (x=0) e dividindo-se pelo volume total de carena. A integral é
        de x * A(x) dx, onde A(x) é a curva de áreas seccionais.
        """
        if self.volume < 1e-6 or not self.interpolador_areas:
            self.lcb = 0.0
            return
            
        # Define a função a ser integrada: x * Area(x).
        funcao_momento_longitudinal = lambda x: x * self.interpolador_areas(x)
        
        # Integra para obter o momento longitudinal do volume.
        momento_long_volume = integrar(funcao_momento_longitudinal, self.x_re, self.x_vante)
        
        # LCB é o momento de volume dividido pelo volume.
        self.lcb = momento_long_volume / self.volume

    def _calcular_momento_vertical_secao(self, x_baliza: float) -> float:
        """
        Calcula o momento de área vertical de uma única seção transversal.

        Este é um método auxiliar para o cálculo do VCB. Ele calcula o primeiro
        momento de área da seção submersa em relação à linha de base (z=0).
        A integral é de z * 2y(z) dz.

        Args:
            x_baliza (float): A posição longitudinal (X) da baliza.

        Returns:
            float: O momento vertical da área da seção [m³].
        """
        # A função a ser integrada é z * largura_total(z) = z * 2y(z).
        funcao_momento = lambda z: z * (2 * self._obter_meia_boca(x_baliza, z))
        
        # Integra de 0 (quilha) até o calado atual.
        momento_vertical = integrar(funcao_momento, 0, self.calado)
        return momento_vertical

    def _calcular_vcb(self):
        """
        Calcula a posição vertical do centro de carena (VCB).

        O VCB é obtido através de uma dupla integração. Primeiro, o momento
        vertical de cada seção transversal é calculado (ver `_calcular_momento_vertical_secao`).
        Em seguida, a curva de momentos verticais (Mv(x)) é integrada ao longo
        do comprimento para obter o momento vertical total do volume. O VCB é
        este momento total dividido pelo volume de carena.
        """
        if self.volume < 1e-6:
            self.vcb = 0.0
            return

        # 1. Calcula o momento vertical para cada baliza física do casco.
        momentos_verticais = {
            x: self._calcular_momento_vertical_secao(x) 
            for x in self.casco.posicoes_balizas
        }
            
        # 2. Cria um interpolador para a curva de momentos verticais (Momento = f(x)).
        x_pontos, momentos_pontos = zip(*sorted(momentos_verticais.items()))
        
        if self.casco.metodo_interp == 'pchip':
            interpolador_momentos = PchipInterpolator(x_pontos, momentos_pontos, extrapolate=False)
        else:
            interpolador_momentos = interp1d(x_pontos, momentos_pontos, kind='linear', bounds_error=False, fill_value=0.0)

        # 3. Integra a curva de momentos ao longo do LWL para obter o momento total do volume.
        momento_total_vertical = integrar(interpolador_momentos, self.x_re, self.x_vante)

        # 4. VCB é o momento vertical total dividido pelo volume.
        self.vcb = momento_total_vertical / self.volume

    def _calcular_momento_inercia_transversal(self):
        """
        Calcula o momento de inércia transversal (I_T) da área do plano de flutuação.

        Este momento de inércia é calculado em relação à linha de centro do navio e é
        fundamental para determinar a estabilidade transversal inicial (BMt). A fórmula
        utilizada é a integral de (2/3) * y(x)³ dx ao longo do LWL, onde y(x) é a
        curva de meia-boca da linha d'água.
        """
        # A validação previne cálculos sem um interpolador definido.
        if not self.interpolador_wl:
            self.momento_inercia_transversal = 0.0
            return

        # Define a função a ser integrada: (2/3) * y(x)³.
        funcao_momento_inercia = lambda x: (2/3) * (self.interpolador_wl(x)**3)
        
        # Integra para obter o momento de inércia transversal total.
        self.momento_inercia_transversal = integrar(funcao_momento_inercia, self.x_re, self.x_vante)

    def _calcular_momento_inercia_longitudinal(self):
        """
        Calcula o momento de inércia longitudinal (I_L) da área do plano de flutuação.

        Este momento de inércia é calculado em relação a um eixo transversal que passa
        pelo centro de flutuação (LCF). É essencial para a estabilidade longitudinal
        e cálculos de trim (BMl, MTc). A fórmula é a integral de (x - LCF)² * 2y(x) dx,
        aplicando o Teorema dos Eixos Paralelos.
        """
        if not self.interpolador_wl:
            self.momento_inercia_longitudinal = 0.0
            return
            
        # Define a função a ser integrada: (distância ao LCF)² * largura_total(x).
        funcao_momento_inercia = lambda x: ((x - self.lcf)**2) * (2 * self.interpolador_wl(x))
        
        # Integra para obter o momento de inércia longitudinal total.
        self.momento_inercia_longitudinal = integrar(funcao_momento_inercia, self.x_re, self.x_vante)

    def _calcular_propriedades_derivadas(self):
        """
        Calcula as propriedades hidrostáticas finais que dependem dos valores base.

        Após o cálculo das áreas, volumes, centros e momentos de inércia, este
        método calcula os parâmetros de estabilidade e os coeficientes de forma,
        que são relações adimensionais usadas para caracterizar a geometria do casco.
        """
        # --- Estabilidade Transversal ---
        # Raio metacêntrico transversal (BMt): I_T / Volume
        self.bmt = self.momento_inercia_transversal / self.volume if self.volume > 1e-6 else 0.0
        # Altura do metacentro transversal acima da quilha (KMt): VCB + BMt
        self.kmt = self.vcb + self.bmt

        # --- Estabilidade Longitudinal ---
        # Raio metacêntrico longitudinal (BMl): I_L / Volume
        self.bml = self.momento_inercia_longitudinal / self.volume if self.volume > 1e-6 else 0.0
        # Altura do metacentro longitudinal acima da quilha (KMl): VCB + BMl
        self.kml = self.vcb + self.bml

        # --- Outras Propriedades Hidrostáticas ---
        # Toneladas por Centímetro de Imersão (TPC): (AWP * densidade) / 100
        self.tpc = (self.area_plano_flutuacao * self.densidade) / 100.0
        
        # Momento para Trimestre em 1 cm (MTc): (I_L * densidade) / (100 * LWL)
        self.mtc = (self.momento_inercia_longitudinal * self.densidade) / (100 * self.lwl) if self.lwl > 1e-6 else 0.0
        
        # --- Coeficientes de Forma ---
        # Área da seção mestra (Am): a maior área de seção transversal calculada.
        area_secao_mestra = max(self.areas_secoes.values()) if self.areas_secoes else 0.0

        # Coeficiente de Bloco (Cb): Volume / (LWL * BWL * T)
        denominador_bloco = self.lwl * self.bwl * self.calado
        self.cb = self.volume / denominador_bloco if denominador_bloco > 1e-6 else 0.0

        # Coeficiente Prismático (Cp): Volume / (Am * LWL)
        denominador_prismatico = area_secao_mestra * self.lwl
        self.cp = self.volume / denominador_prismatico if denominador_prismatico > 1e-6 else 0.0
        
        # Coeficiente do Plano de Flutuação (Cwp): AWP / (LWL * BWL)
        denominador_plano_flutuacao = self.lwl * self.bwl
        self.cwp = self.area_plano_flutuacao / denominador_plano_flutuacao if denominador_plano_flutuacao > 1e-6 else 0.0

        # Coeficiente de Seção Mestra (Cm): Cb / Cp ou Am / (BWL * T)
        self.cm = self.cb / self.cp if self.cp > 1e-6 else 0.0

    def _calcular_todas_propriedades(self):
        """
        Método orquestrador que executa todos os cálculos na ordem correta.
        """
        print(f"\n--- Calculando propriedades para o calado T = {self.calado:.3f} m ---")
        
        # Cálculos de geometria base
        self._calcular_dimensoes_linha_dagua()
        self._calcular_area_plano_flutuacao()
        
        # Cálculo dos centroides longitudinais
        self._calcular_lcf()
        
        # Cálculos de volume (requerem as áreas das seções)
        for x_pos in self.casco.posicoes_balizas:
            self.areas_secoes[x_pos] = self._calcular_area_secao(x_pos)
        self._calcular_volume_deslocamento()
        self._calcular_lcb()

        # Cálculo do centroide vertical (requer o volume)
        self._calcular_vcb()

        # Cálculo dos momentos de inércia
        self._calcular_momento_inercia_transversal()
        self._calcular_momento_inercia_longitudinal()

        # Cálculo das propriedades finais
        self._calcular_propriedades_derivadas()

def calcular_propriedades_para_um_calado(args):
    """
    Função "worker" projetada para ser executada em um processo separado.

    Esta função de nível superior recebe todos os argumentos necessários, instancia
    a classe de cálculo PropriedadesHidrostaticas e retorna os resultados em um
    dicionário. Este formato é necessário para que a função possa ser "serializada"
    (pickled) pelo módulo multiprocessing.

    Args:
        args (tuple): Uma tupla contendo (interpolador_casco, calado, densidade).

    Returns:
        dict: Um dicionário com os resultados hidrostáticos para o calado fornecido.
    """
    interpolador, calado, densidade = args
    # A instanciação da classe executa todos os cálculos para este calado.
    props = PropriedadesHidrostaticas(interpolador, calado, densidade)
    
    # Monta um dicionário com os resultados formatados.
    return {
        'Calado (m)': calado,
        'Volume (m³)': props.volume, 'Desloc. (t)': props.deslocamento,
        'AWP (m²)': props.area_plano_flutuacao, 'LWL (m)': props.lwl, 'BWL (m)': props.bwl,
        'LCB (m)': props.lcb, 'VCB (m)': props.vcb, 'LCF (m)': props.lcf,
        'BMt (m)': props.bmt, 'KMt (m)': props.kmt, 'BMl (m)': props.bml, 'KMl (m)': props.kml,
        'TPC (t/cm)': props.tpc, 'MTc (t·m/cm)': props.mtc, 'Cb': props.cb, 'Cp': props.cp,
        'Cwp': props.cwp, 'Cm': props.cm,
    }

class CalculadoraHidrostatica:
    """
    Orquestra o cálculo das curvas hidrostáticas para múltiplos calados
    utilizando processamento paralelo para otimizar a performance.
    """
    def __init__(self, interpolador_casco: InterpoladorCasco, densidade: float):
        """
        Inicializa a calculadora.

        Args:
            interpolador_casco (InterpoladorCasco): O objeto com a geometria do casco.
            densidade (float): A densidade da água [t/m³].
        """
        self.casco = interpolador_casco
        self.densidade = densidade
        
    def calcular_curvas(self, lista_de_calados: list) -> pd.DataFrame:
        """
        Executa o cálculo hidrostático para uma lista de calados em paralelo.

        Args:
            lista_de_calados (list): Lista de calados a serem calculados.

        Returns:
            pd.DataFrame: Um DataFrame do pandas contendo a tabela de curvas hidrostáticas.
        """
        start_time = time.perf_counter()
        print(f"\n-> Iniciando cálculo PARALELO para {len(lista_de_calados)} calados...")
        
        # Prepara a lista de tarefas. Cada tarefa é uma tupla de argumentos
        # para a nossa função 'worker'.
        tarefas = [(self.casco, calado, self.densidade) for calado in lista_de_calados]
        
        resultados = []
        # ProcessPoolExecutor gerencia um pool de processos (um para cada núcleo de CPU, por padrão),
        # distribuindo as tarefas entre eles.
        with concurrent.futures.ProcessPoolExecutor() as executor:
            # O método 'map' aplica a função 'calcular_propriedades_para_um_calado'
            # a cada item da lista 'tarefas' e coleta os resultados.
            resultados = list(executor.map(calcular_propriedades_para_um_calado, tarefas))
            
        duration = time.perf_counter() - start_time
        print(f"-> Cálculo finalizado em {duration:.2f} segundos.")
        
        # Garante que os resultados estejam ordenados pelo calado.
        resultados.sort(key=lambda r: r['Calado (m)'])
        return pd.DataFrame(resultados)
