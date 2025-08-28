import pandas as pd
import numpy as np
from scipy.interpolate import interp1d, PchipInterpolator
from scipy.optimize import fsolve
from scipy.integrate import quad
from typing import Dict, Any, List

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

        self._calcular_todas_propriedades()

    def _obter_meia_boca(self, x: float, z: float) -> float:
        """
        Obtém a meia-boca (y) para uma dada estação 'x' e altura 'z' existentes.

        Este método consulta diretamente o dicionário de interpoladores para uma estação
        específica. Note que ele não realiza interpolação longitudinal entre estações.

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
            return float(np.nan_to_num(meia_boca))
        else:
            # Se nenhuma função exata para a estação 'x' for encontrada, retorna 0.0.
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
        area_meio_casco, erro_integracao = quad(
            lambda z: self._obter_meia_boca(x_baliza, z), 0, self.calado
        )
        
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
        meia_area, _ = quad(self.interpolador_wl, self.x_re, self.x_vante)
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
        volume_calculado, _ = quad(self.interpolador_areas, self.x_re, self.x_vante)
        self.volume = volume_calculado
        
        # 4. Calcula o deslocamento (massa) a partir do volume e da densidade.
        self.deslocamento = self.volume * self.densidade

    def _calcular_todas_propriedades(self):
        """
        Método orquestrador que executa todos os cálculos na ordem correta.

        A sequência é importante devido às dependências entre as propriedades
        (e.g., o volume depende das áreas seccionais).
        """
        print(f"\n--- Calculando propriedades para o calado T = {self.calado:.3f} m ---")
        
        # 1. Calcula as dimensões fundamentais da linha d'água.
        self._calcular_dimensoes_linha_dagua()
        
        # 2. Calcula a área de cada seção transversal e armazena os resultados.
        #    Este passo é necessário antes de calcular o volume.
        for x_pos in self.casco.posicoes_balizas:
            self.areas_secoes[x_pos] = self._calcular_area_secao(x_pos)
        
        # 3. Com as dimensões e áreas prontas, calcula as propriedades de alto nível.
        self._calcular_area_plano_flutuacao()
        self._calcular_volume_deslocamento()