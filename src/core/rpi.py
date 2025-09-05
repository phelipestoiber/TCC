import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
import copy
import os

import sys
# Adicionar o diretório 'src' ao caminho para encontrar outros módulos
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.core.ch import InterpoladorCasco, PropriedadesHidrostaticas
from src.core.teste import *

class CalculadoraRPI:
    """
    Encapsula todos os cálculos relacionados com o Relatório da Prova de Inclinação.
    """
    def __init__(self, dados_rpi: Dict[str, Any], dados_hidrostaticos: Dict[str, Any], df_hidrostatico: pd.DataFrame, casco_interpolado: InterpoladorCasco):
        """
        Inicializa a calculadora do RPI.

        Args:
            dados_rpi (Dict[str, Any]): O dicionário com os dados da prova de inclinação
                                       recolhidos pelo menu.
            dados_hidrostaticos (Dict[str, Any]): O dicionário com os dados gerais
                                                  da embarcação (Lpp, etc.).
        """
        self.dados_rpi = dados_rpi
        self.dados_hidrostaticos = dados_hidrostaticos
        self.df_hidrostatico = df_hidrostatico
        self.casco = casco_interpolado
        
        # Resultados que serão calculados
        self.calados_nas_marcas: Dict[str, float] = {}
        self.calados_nas_perpendiculares: Dict[str, float] = {}
        self.densidade_media: float = 0.0
        
        # DataFrames para armazenar os detalhes dos pesos
        self.tabela_deducoes = pd.DataFrame()
        self.tabela_acrescimos = pd.DataFrame()

        # Dicionários para os totais
        self.total_deducoes: Dict[str, float] = {}
        self.total_acrescimos: Dict[str, float] = {}

        # Lista para armazenar os dados processados das leituras
        self.leituras_processadas = []

        # Listas para armazenar os momentos calculados
        self.momentos_inclinantes: List[float] = []

        # Características da embarcação
        self.calado_medio: float = 0.0
        self.deflexao: float = 0.0
        self.trim: float = 0.0

        self.hidrostaticos_prova: Dict[str, float] = {}

    def calcular_condicao_flutuacao(self):
        """
        Calcula a condição de flutuação da embarcação (calados nas perpendiculares)
        a partir dos dados de entrada da prova.
        """
        print("\n--- A calcular condição de flutuação da prova ---")
        dados_flutuacao = self.dados_rpi['dados_flutuacao']
        metodo = dados_flutuacao['metodo']

        # --- Parte 1: Obter os calados médios nas marcas de leitura ---
        HMR, HMM, HMV = 0.0, 0.0, 0.0 # Calados nas marcas: Ré, Meio, Vante

        if "bordas livres" in metodo:
            print("-> A calcular calados a partir das bordas livres...")
            # Média das bordas livres em cada ponto
            bl_re = (float(dados_flutuacao['bl_bb_re']) + float(dados_flutuacao['bl_be_re'])) / 2
            bl_meio = (float(dados_flutuacao['bl_bb_meio']) + float(dados_flutuacao['bl_be_meio'])) / 2
            bl_vante = (float(dados_flutuacao['bl_bb_vante']) + float(dados_flutuacao['bl_be_vante'])) / 2

            # Calado = Pontal no local - Borda Livre média
            HMR = float(dados_flutuacao['pontal_re']) - bl_re
            HMMN = float(dados_flutuacao['pontal_meio']) - bl_meio
            HMV = float(dados_flutuacao['pontal_vante']) - bl_vante
        
        else: # "Leitura direta dos calados"
            print("-> A usar calados lidos diretamente...")
            # Assume-se que a banda é desprezível, então a leitura de um bordo é a média.
            HMR = (float(dados_flutuacao['bb_re']) + float(dados_flutuacao['be_re'])) / 2
            HMMN = (float(dados_flutuacao['bb_meio']) + float(dados_flutuacao['be_meio'])) / 2
            HMV = (float(dados_flutuacao['bb_vante']) + float(dados_flutuacao['be_vante'])) / 2
        
        self.calados_nas_marcas = {"re": HMR, "meio": HMM, "vante": HMV}
        print(f"Calados médios nas marcas: Ré={HMR:.4f}m, Meio={HMMN:.4f}m, Vante={HMV:.4f}m")

        # --- Parte 2: Corrigir os calados para as perpendiculares ---
        print("-> A corrigir calados para as perpendiculares...")
        lpp = float(self.dados_hidrostaticos['lpp'])
        LR = float(dados_flutuacao['lr'])
        LM = float(dados_flutuacao['lm'])
        LV = float(dados_flutuacao['lv'])

        # 1. Calcular TM (Trim Medido entre as marcas)
        TM = HMR - HMV

        # 2. Calcular LRV (Distância longitudinal entre as marcas de Ré e Vante)
        LRV = lpp - (LR + LV)
        if LRV <= 0:
            print("ERRO: A distância entre as marcas de calado (LRV) é nula ou negativa. Verifique Lpp, LR e LV.")
            return

        # 3. Calcular tan_theta (tangente do ângulo de trim)
        tan_theta = TM / LRV

        # 4. Correção para a Perpendicular de Ré (HPR)
        dHPR = LR * tan_theta
        HPR = HMR + dHPR

        # 5. Correção para Meio-Navio (HMN)
        dHMN = LM * tan_theta
        HMN = HMMN + dHMN

        # 6. Correção para a Perpendicular de Vante (HPV)
        dHPV = LV * tan_theta
        HPV = HMV - dHPV # NOTA: Usado sinal negativo. Se HMR > HMV (trim pela popa), o calado na PV é menor.

        self.calados_nas_perpendiculares = {"re": HPR, "meio": HMN, "vante": HPV}
        print(f"Calados corrigidos nas perpendiculares: PR={HPR:.4f}m, MN={HMN:.4f}m, PV={HPV:.4f}m")

    def calcular_densidade_media(self):
        """
        Calcula a densidade média da água e processa os itens a deduzir e a acrescentar.
        """
        print("\n--- A calcular densidade e correção de pesos ---")

        # 1. Calcular a densidade média
        densidades = self.dados_rpi['densidades_medidas']
        d_re = float(densidades['re'])
        d_meio = float(densidades['meio'])
        d_vante = float(densidades['vante'])

        self.densidade_media = (d_re + d_meio + d_vante) / 3
        print(f"-> Densidade média da água calculada: {self.densidade_media:.4f} t/m³")

        return self.densidade_media

    def _processar_lista_de_itens(self, lista_itens: List[Dict[str, Any]]) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """
        Método auxiliar genérico para processar uma lista de itens (deduções ou acréscimos).

        Para uma dada lista de itens, ele calcula os momentos longitudinais e verticais
        de cada um, retorna um DataFrame detalhado e um dicionário com os totais.

        Args:
            lista_itens (List[Dict[str, Any]]): A lista de itens a ser processada.

        Returns:
            Tuple[pd.DataFrame, Dict[str, float]]: Um DataFrame com os detalhes e
                                                   um dicionário com os totais.
        """
        if not lista_itens:
            # Se a lista estiver vazia, retorna estruturas vazias
            return pd.DataFrame(), {"peso": 0.0, "momento_long": 0.0, "momento_vert": 0.0}

        # Converte a lista de dicionários num DataFrame do pandas
        df = pd.DataFrame(lista_itens)

        # Converte as colunas para tipos numéricos para garantir que os cálculos funcionam
        df['peso'] = pd.to_numeric(df['peso'])
        df['lcg'] = pd.to_numeric(df['lcg'])
        df['vcg'] = pd.to_numeric(df['vcg'])
        
        # Calcula os momentos para cada item
        df['momento_long'] = df['peso'] * df['lcg']
        df['momento_vert'] = df['peso'] * df['vcg']

        # Calcula os totais somando as colunas do DataFrame
        totais = {
            "peso": df['peso'].sum(),
            "momento_long": df['momento_long'].sum(),
            "momento_vert": df['momento_vert'].sum()
        }
        
        return df, totais

    def calcular_pesos_e_momentos(self):
        """
        Processa as listas de pesos a deduzir e a acrescentar para encontrar os
        seus pesos e momentos totais.
        """
        print("\n--- A calcular densidade e correção de pesos ---")

        # 1. Processar a lista de itens a deduzir
        print("-> A processar itens a deduzir...")
        self.tabela_deducoes, self.total_deducoes = self._processar_lista_de_itens(
            self.dados_rpi['itens_a_deduzir']
        )
        print("Itens a deduzir processados.")

        # 2. Processar a lista de itens a acrescentar
        print("-> A processar itens a acrescentar...")
        self.tabela_acrescimos, self.total_acrescimos = self._processar_lista_de_itens(
            self.dados_rpi['itens_a_acrescentar']
        )
        print("Itens a acrescentar processados.")

    def processar_leituras_inclinacao(self):
        """
        Processa os dados brutos dos pêndulos ou tubos em U para calcular as médias
        das leituras de cada movimento.
        """
        print("\n-> A processar leituras da prova de inclinação...")
        dados_leituras = self.dados_rpi.get('dados_leituras', {})
        metodo = self.dados_rpi.get('metodo_inclinacao', "")

        # Função auxiliar para converter a string "1;2;3;4;5" numa lista de floats
        def _parse_lista_leituras(texto: str) -> List[float]:
            return [float(p.strip()) for p in texto.split(';') if p.strip()]

        if "Pêndulos" in metodo:
            lista_pendulos_brutos = dados_leituras.get('pendulos', [])
            for i, pendulo in enumerate(lista_pendulos_brutos):
                print(f"   A processar Pêndulo nº {i+1}...")
                dados_processados_pendulo = {"tipo": "Pêndulo", "id": i + 1}
                medias_movimentos = []
                for mov_idx, leitura_mov in enumerate(pendulo['leituras']):
                    maximos = _parse_lista_leituras(leitura_mov['maximos'])
                    minimos = _parse_lista_leituras(leitura_mov['minimos'])
                    
                    media_max = np.mean(maximos)
                    media_min = np.mean(minimos)
                    
                    medias_movimentos.append({
                        "movimento": mov_idx,
                        "media_max": media_max,
                        "media_min": media_min
                    })
                dados_processados_pendulo["medias_movimentos"] = medias_movimentos
                self.leituras_processadas.append(dados_processados_pendulo)
        
        elif "Tubos" in metodo:
            lista_tubos_brutos = dados_leituras.get('tubos', [])
            for i, tubo in enumerate(lista_tubos_brutos):
                print(f"   A processar Tubo em U nº {i+1}...")
                dados_processados_tubo = {"tipo": "Tubo em U", "id": i + 1}
                medias_movimentos = []
                for mov_idx, leitura_mov in enumerate(tubo['leituras']):
                    max_bb = _parse_lista_leituras(leitura_mov['maximos_bb'])
                    min_bb = _parse_lista_leituras(leitura_mov['minimos_bb'])
                    max_be = _parse_lista_leituras(leitura_mov['maximos_be'])
                    min_be = _parse_lista_leituras(leitura_mov['minimos_be'])

                    medias_movimentos.append({
                        "movimento": mov_idx,
                        "media_max_bb": np.mean(max_bb),
                        "media_min_bb": np.mean(min_bb),
                        "media_max_be": np.mean(max_be),
                        "media_min_be": np.mean(min_be),
                    })
                dados_processados_tubo["medias_movimentos"] = medias_movimentos
                self.leituras_processadas.append(dados_processados_tubo)
        
        print("-> Leituras processadas com sucesso.")

    def calcular_momentos_inclinantes(self):
        """
        Calcula a série de momentos inclinantes transversais para os 8 movimentos
        da prova de inclinação.
        """
        print("\n-> A calcular momentos inclinantes para cada movimento...")
        
        # Filtrar apenas os pesos da prova da lista de deduções
        pesos_prova_brutos = [
            item for item in self.dados_rpi['itens_a_deduzir']
            if 'Peso da prova' in item['nome']
        ]

        # Se não houver pesos sólidos (ex: prova com lastro), não há nada a fazer.
        if not pesos_prova_brutos:
            print("   Prova com pesos líquidos. A saltar cálculo de momentos de pesos sólidos.")
            self.momentos_inclinantes = [0.0] * 9 # Retorna uma lista de zeros
            return

        # Converter os dados para tipos numéricos
        pesos_prova = []
        for p in pesos_prova_brutos:
            pesos_prova.append({
                'nome': p['nome'],
                'peso': float(p['peso']),
                'tcg': float(p['tcg'])
            })

        # --- Início da Simulação ---
        # A sequência de movimentos pressupõe uma ordem específica dos pesos.
        # Vamos assumir que foram inseridos na ordem A, B, C, D.
        # Nomenclatura para clareza: A=0, B=1, C=2, D=3
        
        # Usamos deepcopy para não modificar o estado original dos pesos
        estado_atual_pesos = copy.deepcopy(pesos_prova)
        momentos_calculados = []

        # Função auxiliar para calcular o momento total num dado estado
        def _get_momento_total(estado_dos_pesos):
            momento_total = 0
            for p in estado_dos_pesos:
                momento_total += p['peso'] * p['tcg']
            return momento_total

        # Movimento 0: Estado Inicial
        momento_inicial = _get_momento_total(estado_atual_pesos)
        momentos_calculados.append(momento_inicial)
        print(f"   Movimento 0 (Inicial): Momento Total = {momento_inicial:.4f} t.m")

        # Sequência de movimentos [ (índice_do_peso, 'direção'), ... ]
        # A sua sequência é: B, C, C(volta), B(volta), A, D, A(volta), D(volta)
        sequencia_movimentos = [1, 2, 2, 1, 0, 3, 0, 3]

        for i, idx_peso in enumerate(sequencia_movimentos):
            # Inverte o sinal do TCG do peso a ser movido
            estado_atual_pesos[idx_peso]['tcg'] *= -1
            
            momento_movimento = _get_momento_total(estado_atual_pesos)
            momentos_calculados.append(momento_movimento)
            print(f"   Movimento {i+1} ({estado_atual_pesos[idx_peso]['nome']}): Momento Total = {momento_movimento:.4f} t.m")
            
        self.momentos_inclinantes = momentos_calculados
        print("-> Cálculo de momentos inclinantes concluído.")

    def calcular_caracteristicas_hidrostaticas_prova(self):
        """
        Calcula o calado médio, deflexão e trim da embarcação na condição da prova.

        Utiliza os calados nas marcas e nas perpendiculares calculados anteriormente.
        """
        print("\n-> A calcular características hidrostáticas da prova...")
        
        # Certificar que o cálculo prévio foi executado
        if not self.calados_nas_perpendiculares or not self.calados_nas_marcas:
            print("ERRO: É necessário primeiro calcular a condição de flutuação.")
            return

        HPR = self.calados_nas_perpendiculares["re"]
        HPV = self.calados_nas_perpendiculares["vante"]
        HMN = self.calados_nas_perpendiculares["meio"]
        
        HMR = self.calados_nas_marcas["re"]
        HMV = self.calados_nas_marcas["vante"]

        # 1. Calcular o Calado Médio (TM)
        self.calado_medio = (HPR + HPV) / 2
        print(f"   Calado Médio (nas PP) calculado: {self.calado_medio:.4f} m")

        # 2. Calcular a Deflexão (Hogging/Sagging)
        self.deflexao = HMN - self.calado_medio
        deflexao_tipo = "Hogging (Alquebramento)" if self.deflexao < 0 else "Sagging (Contra-alquebramento)"
        print(f"   Deflexão calculada: {abs(self.deflexao):.4f} m ({deflexao_tipo})")

        # 3. Calcular o Trim (t)
        self.trim = HMR - HMV
        trim_direcao = "Trim pela Popa" if self.trim > 0 else "Trim pela Proa" if self.trim < 0 else "Sem Trim"
        print(f"   Trim (nas marcas) calculado: {abs(self.trim):.4f} m ({trim_direcao})")



    def aplicar_correcao_deflexao(self) -> InterpoladorCasco:
        """
        Aplica a correção de deflexão (hogging/sagging) à geometria do casco.

        Este método utiliza a deflexão máxima calculada a meio-navio e a modela
        como uma curva parabólica ao longo do comprimento da embarcação. A correção
        vertical correspondente é então aplicada a cada ponto da tabela de cotas
        original, gerando um novo objeto InterpoladorCasco que representa a
        geometria deformada da embarcação.

        Returns:
            InterpoladorCasco: Um novo objeto com a geometria do casco corrigida
                               para a deflexão.
        """
        print("\n-> A aplicar correção de deflexão à tabela de cotas...")
        
        if self.deflexao is None or self.casco is None:
            print("ERRO: A deflexão e a geometria do casco devem ser calculadas primeiro.")
            return None

        deflexao_maxima = self.deflexao
        lpp = float(self.dados_hidrostaticos['lpp'])
        
        # Copia a tabela de cotas original para não modificar os dados base
        tabela_corrigida = self.casco.tabela_cotas.copy()
        
        # 1. Função da parábola de deflexão
        # d(x) = 4 * deflexao_maxima * (Lpp*x - x^2) / Lpp^2
        def calcular_delta_z(x):
            if lpp == 0: return 0
            return (4 * deflexao_maxima * (lpp * x - x**2)) / (lpp**2)

        # 2. Aplicar a correção a todos os pontos 'z' da tabela
        # A função .apply do pandas é eficiente para esta operação
        tabela_corrigida['z'] = tabela_corrigida.apply(
            lambda row: row['z'] + calcular_delta_z(row['x']),
            axis=1
        )
        
        print("   Correção de deflexão aplicada com sucesso.")
        
        # Cria e retorna um novo interpolador com a geometria corrigida
        casco_corrigido = InterpoladorCasco(
            tabela_corrigida,
            metodo_interp=self.casco.metodo_interp
        )
        return casco_corrigido

    def calcular_hidrostaticos_corrigidos(self):
        """
        Calcula as propriedades hidrostáticas corrigidas para trim e deflexão.

        Este método orquestra a utilização das classes do módulo 'ch' para
        aplicar as correções e obter as propriedades finais da embarcação
        na condição "como inclinado".
        """
        print("\n-> Calculando propriedades hidrostáticas corrigidas para trim e deflexão...")

        # 1. Aplicar a correção de deflexão (Hogging/Sagging)
        #    Instanciamos PropriedadesDeflexao e passamos para um novo InterpoladorCasco.
        #    O próprio InterpoladorCasco irá gerar a geometria corrigida.
        prop_deflexao = PropriedadesDeflexao(
            deflexao=self.deflexao,
            tabela_cotas=self.casco.tabela_cotas,  # Usa a tabela de cotas original
            lpp=self.dados_hidrostaticos['lpp']
        )
        
        casco_corrigido = InterpoladorCasco(
            tabela_cotas=self.casco.tabela_cotas, # Tabela original como base
            metodo_interp=self.casco.metodo_interp,
            prop_deflexao=prop_deflexao # O objeto de deflexão aplica a correção
        )
        print(f"   - Geometria do casco corrigida para deflexão de {self.deflexao*1000:.1f} mm.")

        # 2. Preparar os dados de trim
        #    Usamos os calados corrigidos (já calculados em um passo anterior)
        #    para instanciar a classe PropriedadesTrim.
        prop_trim = PropriedadesTrim(
            calado_re=self.calados_nas_perpendiculares['re'],
            calado_vante=self.calados_nas_perpendiculares['vante'],
            lpp=self.dados_hidrostaticos['lpp'],
            posicoes_balizas=casco_corrigido.posicoes_balizas
        )
        print(f"   - Condição de trim aplicada: Tr={prop_trim.calado_re:.3f}m, Tv={prop_trim.calado_vante:.3f}m.")

        # 3. Calcular as propriedades hidrostáticas finais
        #    Instanciamos a calculadora principal, passando o casco corrigido
        #    e as propriedades de trim.
        self.propriedades_hidrostaticas_corrigidas = PropriedadesHidrostaticas(
            interpolador_casco=casco_corrigido,
            densidade=self.densidade_media,
            prop_trim=prop_trim
        )
        
        # 2. Extrair e armazenar todos os resultados necessários
        props = self.propriedades_hidrostaticas_corrigidas
        self.hidrostaticos_prova = {
            "Deslocamento": props.deslocamento,
            "Volume": props.volume,
            "LCB": props.lcb,
            "VCB": props.vcb,
            "KMt": props.kmt,
            "MTc": props.mtc
        }

        print("   Propriedades na condição da prova obtidas com sucesso:")
        for chave, valor in self.hidrostaticos_prova.items():
            print(f"     - {chave}: {valor:.4f}")


if __name__ == '__main__':
    import os
    import sys
    # Adicionar o diretório 'src' ao caminho para encontrar outros módulos
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from src.core.teste import InterpoladorCasco

    print("--- INICIANDO TESTE DE CORREÇÃO DE DEFLEXÃO ---")

    # 1. Preparar dados de teste (simulando a execução do programa)
    # Carregar a geometria original
    diretorio_raiz_projeto = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    caminho_csv = os.path.join(diretorio_raiz_projeto, "data", "exemplos_tabelas_cotas", "TABELA DE COTAS.csv")
    tabela_cotas_original = pd.read_csv(caminho_csv)
    tabela_cotas_original.columns = [col.strip().lower() for col in tabela_cotas_original.columns]
    
    casco_original = InterpoladorCasco(tabela_cotas_original, metodo_interp='linear')

    # Simular dados de entrada que seriam recolhidos pelo menu
    dados_hidro_teste = {'lpp': 19.713} # Lpp da embarcação de exemplo

    # Simular dados de entrada do RPI que seriam recolhidos pelo menu
    # Usamos calados lidos para simular um trim pela popa e um sagging
    dados_rpi_teste = {
        'dados_flutuacao': {
            'metodo': 'Leitura direta dos calados', 'lr': 0, 'lm': 0, 'lv': 0,
            'bb_re': '1.875', 'bb_meio': '1.835', 'bb_vante': '1.775', 'be_re': '1.875', 'be_meio': '1.835', 'be_vante': '1.775'
        },
        'densidades_medidas': {'re': '1.025', 'meio': '1.025', 'vante': '1.025'}
    }
    
    # Criar uma instância da calculadora
    # Passamos um DataFrame vazio para df_hidrostatico, pois não é usado neste fluxo
    calculadora = CalculadoraRPI(dados_rpi_teste, dados_hidro_teste, pd.DataFrame(), casco_original)
    
    # 2. Executar os cálculos em sequência
    calculadora.calcular_condicao_flutuacao()
    calculadora.calcular_caracteristicas_hidrostaticas_prova()
    
    # Corrigir a deflexão para obter a nova geometria
    casco_corrigido = calculadora.aplicar_correcao_deflexao()

    # Calcular os hidrostáticos finais usando o casco corrigido
    calculadora.calcular_densidade_media()
    calculadora.calcular_hidrostaticos_corrigidos()

    # 3. Verificar os resultados (já são impressos dentro do método)
    print("\n--- TESTE CONCLUÍDO ---")
