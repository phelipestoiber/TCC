# src/core/rpi.py

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

        self.resultados_inclinacao = []
        self.gm_prova: float = 0.0
        self.vcg_prova: float = 0.0

        self.navio_leve_resultados: Dict[str, float] = {}

        self.hidrostaticos_navio_leve: Dict[str, float] = {}
        self.flutuacao_navio_leve: Dict[str, float] = {}

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
                        "media_min": media_min,
                        "media": (media_max + media_min) / 2
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
                        "media_bb": np.mean(max_bb + min_bb),
                        "media_max_be": np.mean(max_be),
                        "media_min_be": np.mean(min_be),
                        "media_be": np.mean(max_be + min_be),
                    })
                dados_processados_tubo["medias_movimentos"] = medias_movimentos
                self.leituras_processadas.append(dados_processados_tubo)
        print(f'leituras processadas: {self.leituras_processadas}')
        
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
            "MTc": props.mtc,
            "LCG": props.lcb - (100 * props.mtc * prop_trim.trim) / props.deslocamento
        }

        print("   Propriedades na condição da prova obtidas com sucesso:")
        for chave, valor in self.hidrostaticos_prova.items():
            print(f"     - {chave}: {valor:.4f}")

    def calcular_gm_vcg(self):
        """
        Calcula a altura metacêntrica (GM) para cada movimento, o GM médio
        e a posição final do centro de gravidade (VCG/KG) da embarcação
        na condição "como inclinada".
        """
        print("\n--- A calcular GM e VCG a partir dos dados da prova ---")

        # 1. Validar se os dados necessários estão disponíveis
        if not self.leituras_processadas:
            print("ERRO: As leituras da inclinação não foram processadas.")
            return
        if not self.momentos_inclinantes or len(self.momentos_inclinantes) < 9:
            print("ERRO: Os momentos inclinantes não foram calculados.")
            return
        if not self.hidrostaticos_prova or "Deslocamento" not in self.hidrostaticos_prova:
            print("ERRO: O deslocamento corrigido não foi calculado.")
            return

        deslocamento = self.hidrostaticos_prova["Deslocamento"]
        momento_inicial = self.momentos_inclinantes[0]
        
        # Estrutura para obter o comprimento/distância de cada dispositivo
        comprimentos_dispositivos = []
        metodo = self.dados_rpi.get('metodo_inclinacao', "")
        if "Pêndulos" in metodo:
            pendulos_info = self.dados_rpi['dados_leituras']['pendulos']
            comprimentos_dispositivos = [p['comprimento'] for p in pendulos_info]
        elif "Tubos" in metodo:
            tubos_info = self.dados_rpi['dados_leituras']['tubos']
            comprimentos_dispositivos = [t['distancia_vertical'] for t in tubos_info]

        lista_gm_movimentos = []

        # Itera sobre os 8 movimentos (índices 1 a 8)
        for mov_idx in range(1, 9):
            # Ignorar movimentos 4 e 8 conforme a sua lógica (índices 3 e 7)
            if mov_idx in [4, 8]:
                continue
                
            tangentes_do_movimento = []

            # Para cada movimento, calcula a tangente em cada dispositivo
            for disp_idx, dispositivo in enumerate(self.leituras_processadas):
                leituras_mov = dispositivo["medias_movimentos"]
                leitura_inicial = leituras_mov[0]
                leitura_atual = leituras_mov[mov_idx]
                
                comprimento_L = comprimentos_dispositivos[disp_idx]
                if comprimento_L == 0: continue # Evita divisão por zero

                deflexao = 0.0
                # Passo 1: Calcular a deflexão
                if dispositivo['tipo'] == 'Pêndulo':
                    deflexao = leitura_atual['media'] - leitura_inicial['media']
                
                elif dispositivo['tipo'] == 'Tubo em U':
                    # deflexão = (L1 - L2) + (L4 - L3)
                    l1 = leitura_inicial['media_bb']
                    l2 = leitura_atual['media_bb']
                    l3 = leitura_inicial['media_be']
                    l4 = leitura_atual['media_be']
                    deflexao = (l1 - l2) + (l4 - l3)

                # Passo 2: Calcular a tangente de theta
                tan_theta = deflexao / comprimento_L
                tangentes_do_movimento.append(tan_theta)

            # Passo 3: Calcular a tangente média do movimento
            if not tangentes_do_movimento: continue
            tan_media_movimento = np.mean(tangentes_do_movimento)
            
            # Passo 4: Calcular a altura metacêntrica (GM) do movimento
            # O momento inclinante é a *diferença* em relação ao estado inicial
            momento_inclinante = self.momentos_inclinantes[mov_idx] - momento_inicial
            
            # Evita divisão por zero se a tangente for nula
            if abs(tan_media_movimento * deslocamento) < 1e-9:
                gm_movimento = 0.0
            else:
                gm_movimento = abs(momento_inclinante / (tan_media_movimento * deslocamento))
            
            lista_gm_movimentos.append(gm_movimento)

            # Armazena os resultados detalhados
            self.resultados_inclinacao.append({
                "Movimento": mov_idx,
                "Momento Inclinante (t.m)": momento_inclinante,
                "Tangente Média": tan_media_movimento,
                "GM Calculado (m)": gm_movimento
            })
            print(f"   Movimento {mov_idx}: Mom. Inclinante={momento_inclinante:+.2f} t.m, "
                  f"Tan(θ)={tan_media_movimento:.5f}, GM={gm_movimento:.4f} m")

        # Calcular o GM final como a média dos GMs válidos
        if not lista_gm_movimentos:
            print("AVISO: Nenhum GM pôde ser calculado. Verifique os dados de entrada.")
            return

        self.gm_prova = np.mean(lista_gm_movimentos)
        print(f"\n-> Altura Metacêntrica (GM) média calculada: {self.gm_prova:.4f} m")

        # Calcular o VCG/KG final
        kmt_prova = self.hidrostaticos_prova.get("KMt", 0.0)
        self.vcg_prova = kmt_prova - self.gm_prova
        print(f"   KMt na condição da prova: {kmt_prova:.4f} m")
        print(f"-> VCG (KG) na condição 'como inclinado': {self.vcg_prova:.4f} m")

    def calcular_condicao_navio_leve(self):
        """
        Calcula as características finais da embarcação na condição de navio leve.

        Este método parte da condição "como inclinado" e aplica as correções
        dos pesos a deduzir e a acrescentar para chegar ao deslocamento e
        posição do centro de gravidade (LCG e VCG) finais.
        """
        print("\n--- A calcular a condição final de Navio Leve ---")

        # 1. Validar se os dados de partida estão disponíveis
        if not self.hidrostaticos_prova or not self.vcg_prova:
            print("ERRO: A condição 'como inclinado' não foi calculada. Execute os passos anteriores.")
            return

        # 2. Obter os dados da condição "como inclinado"
        desloc_prova = self.hidrostaticos_prova['Deslocamento']
        lcg_prova = self.hidrostaticos_prova['LCG']
        vcg_prova = self.vcg_prova

        # Calcular os momentos iniciais
        momento_long_prova = desloc_prova * lcg_prova
        momento_vert_prova = desloc_prova * vcg_prova

        print(f"   Condição 'Como Inclinado':")
        print(f"   - Deslocamento: {desloc_prova:.4f} t")
        print(f"   - LCG: {lcg_prova:.4f} m, VCG: {vcg_prova:.4f} m")

        # 3. Obter os totais dos pesos a deduzir e a acrescentar
        #    Esses valores já foram calculados pelo método 'calcular_pesos_e_momentos'
        deducoes = self.total_deducoes
        acrescimos = self.total_acrescimos
        
        print(f"   - Total a Deduzir: {deducoes['peso']:.4f} t")
        print(f"   - Total a Acrescentar: {acrescimos['peso']:.4f} t")

        # 4. Calcular os totais para a condição de Navio Leve
        #    Deslocamento Leve = Desloc. Prova - Deduções + Acréscimos
        desloc_leve = desloc_prova - deducoes['peso'] + acrescimos['peso']
        
        #    Momento Long. Leve = Mom. Long. Prova - Mom. Long. Deduções + Mom. Long. Acréscimos
        momento_long_leve = momento_long_prova - deducoes['momento_long'] + acrescimos['momento_long']
        
        #    Momento Vert. Leve = Mom. Vert. Prova - Mom. Vert. Deduções + Mom. Vert. Acréscimos
        momento_vert_leve = momento_vert_prova - deducoes['momento_vert'] + acrescimos['momento_vert']

        # 5. Calcular o LCG e VCG finais
        if abs(desloc_leve) < 1e-6:
            print("ERRO: Deslocamento de navio leve é zero. Impossível calcular LCG e VCG.")
            lcg_leve = 0.0
            vcg_leve = 0.0
        else:
            lcg_leve = momento_long_leve / desloc_leve
            vcg_leve = momento_vert_leve / desloc_leve

        # 6. Armazenar os resultados
        self.navio_leve_resultados = {
            'Deslocamento Leve (t)': desloc_leve,
            'Momento Long. Leve (t.m)': momento_long_leve,
            'Momento Vert. Leve (t.m)': momento_vert_leve,
            'LCG Leve (m)': lcg_leve,
            'VCG Leve (m)': vcg_leve
        }

        print("\n-> Resultados Finais para Navio Leve:")
        for chave, valor in self.navio_leve_resultados.items():
            print(f"   - {chave}: {valor:.4f}")

    def calcular_hidrostaticos_navio_leve(self):
        """
        Calcula a condição de flutuação (trim e calado) e as características
        hidrostáticas para a condição de navio leve através de um processo iterativo.
        """
        print("\n--- A calcular hidrostáticas para a condição de Navio Leve (processo iterativo) ---")

        # 1. Obter os alvos: Deslocamento e LCG do navio leve
        if not self.navio_leve_resultados:
            print("ERRO: A condição de navio leve deve ser calculada primeiro.")
            return
        
        desloc_alvo = self.navio_leve_resultados['Deslocamento Leve (t)']
        lcg_alvo = self.navio_leve_resultados['LCG Leve (m)']
        lpp = self.dados_hidrostaticos['lpp']

        # 2. Iniciar o processo iterativo
        # Começamos com uma estimativa inicial: o calado médio da prova, sem trim.
        calado_re_atual = self.calado_medio
        calado_vante_atual = self.calado_medio
        
        max_iteracoes = 100
        tolerancia = 1e-4 # Tolerância para convergência (0.01%)

        for i in range(max_iteracoes):
            print(f"\nIteração {i+1}:")
            print(f"   - Tentativa: Calado Ré={calado_re_atual:.4f}m, Calado Vante={calado_vante_atual:.4f}m")

            # 3. Calcular as hidrostáticas para a tentativa atual
            prop_trim_iter = PropriedadesTrim(
                calado_re=calado_re_atual,
                calado_vante=calado_vante_atual,
                lpp=lpp,
                posicoes_balizas=self.casco.posicoes_balizas
            )
            props_iter = PropriedadesHidrostaticas(
                interpolador_casco=self.casco, # Usamos o casco original (sem deflexão da prova)
                densidade=self.densidade_media,
                prop_trim=prop_trim_iter
            )
            
            desloc_calc = props_iter.deslocamento
            lcb_calc = props_iter.lcb
            mtc_calc = props_iter.mtc
            lcf_calc = props_iter.lcf
            
            # 4. Verificar a convergência
            erro_desloc = (desloc_calc - desloc_alvo) / desloc_alvo if desloc_alvo else 0
            erro_lcg = (lcb_calc - lcg_alvo) / lpp if lpp else 0

            print(f"   - Resultados: Desloc={desloc_calc:.3f}t (erro {erro_desloc:+.4%}), "
                  f"LCB={lcb_calc:.3f}m (erro LCG {erro_lcg:+.4%})")

            if abs(erro_desloc) < tolerancia and abs(erro_lcg) < tolerancia:
                print(f"\n-> Convergência alcançada!, iteração {i}")
                # Armazena os resultados finais
                self.flutuacao_navio_leve = {
                    'Calado na PP de Ré (m)': calado_re_atual,
                    'Calado na PP de Vante (m)': calado_vante_atual,
                    'Calado Médio (m)': (calado_re_atual + calado_vante_atual) / 2,
                    'Trim (m)': calado_vante_atual - calado_re_atual
                }
                self.hidrostaticos_navio_leve = {
                    'Deslocamento (t)': props_iter.deslocamento,
                    'LCB (m)': props_iter.lcb,
                    'MTC (t.m/cm)': props_iter.mtc,
                    'LCF (m)': props_iter.lcf,
                    'KMt (m)': props_iter.kmt
                }
                return # Termina o método com sucesso

            # 5. Se não convergiu, ajustar os calados para a próxima iteração
            # Correção do Trim para acertar o LCG:
            momento_trimante = desloc_alvo * (lcg_alvo - lcb_calc)
            trim_necessario = momento_trimante / (mtc_calc * 100) if mtc_calc else 0
            
            # Distribui o trim em relação ao LCF para encontrar os novos calados de ré e vante
            calado_re_sem_desloc_corr = calado_re_atual - trim_necessario * (lcf_calc / lpp)
            calado_vante_sem_desloc_corr = calado_vante_atual + trim_necessario * (1 - (lcf_calc / lpp))
            
            # Correção do Calado Médio para acertar o Deslocamento:
            # Re-calcula as props na nova condição de trim para obter um TPC mais preciso
            tpc_iter = props_iter.tpc
            correcao_calado_medio = (desloc_alvo - desloc_calc) / (tpc_iter * 100) if tpc_iter else 0
            
            calado_re_atual = calado_re_sem_desloc_corr + correcao_calado_medio
            calado_vante_atual = calado_vante_sem_desloc_corr + correcao_calado_medio

        print("AVISO: O cálculo de hidrostáticas para navio leve não convergiu após o número máximo de iterações.")


if __name__ == '__main__':
    import os
    import sys
    # Adicionar o diretório 'src' ao caminho para encontrar outros módulos
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    # Assegure-se que as classes corretas estão sendo importadas
    from src.core.teste import InterpoladorCasco

    print("--- INICIANDO TESTE COMPLETO DO CÁLCULO DE RPI ---")

    # 1. Preparar dados de teste
    diretorio_raiz_projeto = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    caminho_csv = os.path.join(diretorio_raiz_projeto, "data", "exemplos_tabelas_cotas", "TABELA DE COTAS.csv")
    tabela_cotas_original = pd.read_csv(caminho_csv)
    tabela_cotas_original.columns = [col.strip().lower() for col in tabela_cotas_original.columns]
    casco_original = InterpoladorCasco(tabela_cotas_original, metodo_interp='linear')
    dados_hidro_teste = {'lpp': 19.713} 

    # --- DADOS DE TESTE PARA NAVIO LEVE ---
    dados_rpi_teste = {
        'metodo_inclinacao': 'Tubos em U',
        'dados_flutuacao': {
            'metodo': 'Leitura direta dos calados', 'lr': 0, 'lm': 0, 'lv': 0,
            'bb_re': '1.875', 'bb_meio': '1.835', 'bb_vante': '1.775', 
            'be_re': '1.875', 'be_meio': '1.835', 'be_vante': '1.775'
        },
        'densidades_medidas': {'re': '1.025', 'meio': '1.025', 'vante': '1.025'},
        'dados_leituras': {
            'tubos': [
                {'distancia_vertical': 3.0}, {'distancia_vertical': 3.0}
            ]
        },
        # Injetando pesos de exemplo que seriam coletados pelo menu
        'itens_a_acrescentar': [], # Sem deduções no seu exemplo
        'itens_a_deduzir': [
            {'nome': 'peso 1', 'peso': '0.2', 'lcg': '6.0565', 'vcg': '4.1', 'tcg': '2.75'},
            {'nome': 'peso 2', 'peso': '0.2', 'lcg': '6.2565', 'vcg': '4.1', 'tcg': '-2.75'},
            {'nome': 'peso 3', 'peso': '0.2', 'lcg': '0.8965', 'vcg': '4.6', 'tcg': '-2.20'},
            {'nome': 'peso 4', 'peso': '0.2', 'lcg': '0.8965', 'vcg': '4.6', 'tcg': '2.20'},
            {'nome': 'pessoas', 'peso': '0.32', 'lcg': '6.8565', 'vcg': '4.3', 'tcg': '0'}
        ]
    }
    
    # Criar a instância da calculadora
    calculadora = CalculadoraRPI(dados_rpi_teste, dados_hidro_teste, pd.DataFrame(), casco_original)
    
    # 2. Executar a sequência completa de cálculos
    calculadora.calcular_condicao_flutuacao()
    calculadora.calcular_caracteristicas_hidrostaticas_prova()
    calculadora.calcular_densidade_media()
    calculadora.calcular_hidrostaticos_corrigidos()

    # --- INJEÇÃO DE DADOS PARA GM/VCG ---
    calculadora.momentos_inclinantes = [0, 1.1, 1.98, 1.1, 0, -1.1, -1.98, -1.1, 0]
    leituras_re_bb = [1.24, 1.2255, 1.205, 1.2255, 0, 1.2575, 1.2675, 1.2575, 0]
    leituras_re_be = [1.195, 1.223, 1.245, 1.223, 0, 1.195, 1.165, 1.195, 0]
    leituras_vante_bb = [1.665, 1.645, 1.635, 1.645, 0, 1.685, 1.705, 1.685, 0]
    leituras_vante_be = [1.545, 1.565, 1.585, 1.565, 0, 1.535, 1.51, 1.5355, 0]
    calculadora.leituras_processadas = [
        {"tipo": "Tubo em U", "id": 1, "medias_movimentos": [{"movimento": i, "media_bb": leituras_re_bb[i], "media_be": leituras_re_be[i]} for i in range(9)]},
        {"tipo": "Tubo em U", "id": 2, "medias_movimentos": [{"movimento": i, "media_bb": leituras_vante_bb[i], "media_be": leituras_vante_be[i]} for i in range(9)]}
    ]
    
    calculadora.calcular_gm_vcg()

    # --- CÁLCULO FINAL DE NAVIO LEVE ---
    # Processa os pesos que injetamos no dicionário 'dados_rpi_teste'
    calculadora.calcular_pesos_e_momentos()

    # Executa o cálculo da condição de navio leve
    calculadora.calcular_condicao_navio_leve()

    # Encontra a flutuação e as hidrostáticas para a condição de navio leve
    calculadora.calcular_hidrostaticos_navio_leve()

    # Imprime os resultados finais
    print("\n--- RESULTADOS FINAIS HIDROSTÁTICOS (NAVIO LEVE) ---")
    if calculadora.flutuacao_navio_leve:
        for chave, valor in calculadora.flutuacao_navio_leve.items():
            print(f"   - {chave}: {valor:.4f}")
        for chave, valor in calculadora.hidrostaticos_navio_leve.items():
            print(f"   - {chave}: {valor:.4f}")
            
    print("\n--- TESTE CONCLUÍDO ---")

