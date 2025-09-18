# src/core/eed.py

from typing import List, Dict, Any
import pandas as pd
import logging
import math
import numpy as np
from .ch import PropriedadesTrim, PropriedadesHidrostaticas, InterpoladorCasco
from .cc import CalculadoraCurvasCruzadas


# 1. Configuração do Logging
# Configura o logger para exibir mensagens de nível INFO ou superior.
# Para ver mensagens de DEBUG, mude level=logging.DEBUG
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class CalculadoraEED:
    """
    Realiza os cálculos do Estudo de Estabilidade Definitivo para
    diferentes condições de carregamento.
    """

    # Define as 6 condições de carregamento de forma estruturada
    CONDICOES_CARREGAMENTO = [
        {
            "nome": "Condição 1: Viagem com Carga Plena (100% Consumíveis)",
            "percentuais": {
                "EMBARCAÇÃO LEVE": 1.0,
                "CARGAS": 1.0, "PASSAGEIROS E BAGAGENS": 1.0,
                "TRIPULAÇÃO": 1.0, "GENEROS E ÓLEOS": 1.0
            }
        },
        {
            "nome": "Condição 2: Fim de Viagem com Carga Plena (10% Consumíveis)",
            "percentuais": {
                "EMBARCAÇÃO LEVE": 1.0,
                "CARGAS": 1.0, "PASSAGEIROS E BAGAGENS": 1.0,
                "TRIPULAÇÃO": 1.0, "GENEROS E ÓLEOS": 0.1
            }
        },
        {
            "nome": "Condição 3: Chegada ao Porto sem Carga (100% Consumíveis)",
            "percentuais": {
                "EMBARCAÇÃO LEVE": 1.0,
                "CARGAS": 0.0, "PASSAGEIROS E BAGAGENS": 1.0,
                "TRIPULAÇÃO": 1.0, "GENEROS E ÓLEOS": 1.0
            }
        },
        {
            "nome": "Condição 4: Partida do Porto sem Carga (10% Consumíveis)",
            "percentuais": {
                "EMBARCAÇÃO LEVE": 1.0,
                "CARGAS": 0.0, "PASSAGEIROS E BAGAGENS": 1.0,
                "TRIPULAÇÃO": 1.0, "GENEROS E ÓLEOS": 0.1
            }
        },
        {
            "nome": "Condição 5: Viagem sem Passageiros (100% Consumíveis)",
            "percentuais": {
                "EMBARCAÇÃO LEVE": 1.0,
                "CARGAS": 1.0, "PASSAGEIROS E BAGAGENS": 0.0,
                "TRIPULAÇÃO": 1.0, "GENEROS E ÓLEOS": 1.0
            }
        },
        {
            "nome": "Condição 6: Fim de Viagem sem Passageiros (10% Consumíveis)",
            "percentuais": {
                "EMBARCAÇÃO LEVE": 1.0,
                "CARGAS": 1.0, "PASSAGEIROS E BAGAGENS": 0.0,
                "TRIPULAÇÃO": 1.0, "GENEROS E ÓLEOS": 0.1
            }
        }
    ]

    def __init__(self, dados_estabilidade: Dict[str, Any], deslocamento_leve: float,
                 casco: InterpoladorCasco, dados_hidrostaticos: Dict[str, Any],
                 df_hidrostatico: pd.DataFrame, densidade: float):
        """
        Inicializa a calculadora com todos os dados necessários.

        Args:
            dados_estabilidade (Dict): Dicionário do menu com pesos, tanques, etc.
            deslocamento_leve (float): Deslocamento da embarcação leve (t).
            casco (InterpoladorCasco): O objeto do casco já interpolado.
            dados_hidrostaticos (Dict): Dicionário com dados básicos como 'lpp'.
            densidade (float): A densidade da água para os cálculos.
        """
        logging.info("Instanciando a CalculadoraEED.")

        if deslocamento_leve <= 0:
            logging.error("Deslocamento leve deve ser um valor positivo.")
            raise ValueError("Deslocamento leve inválido.")
        logging.info("Instanciando a CalculadoraEED com dados completos.")
        self.tabela_pesos = dados_estabilidade.get("tabela_pesos", [])
        self.ponto_alagamento = dados_estabilidade.get("ponto_alagamento", {})
        self.dados_tanques = dados_estabilidade.get("dados_tanques", [])
        self.dados_passageiros = dados_estabilidade.get("dados_passageiros", [])
        
        self.deslocamento_leve = deslocamento_leve
        self.casco = casco
        self.densidade = densidade
        self.dados_hidrostaticos = dados_hidrostaticos
        self.df_hidrostatico = df_hidrostatico
        
        self.resultados_condicoes = {}

        # Converte para float e armazena os novos dados
        self.vo_knots = float(dados_estabilidade.get("velocidade_operacao", 0.0))

        # Constantes padrão
        self.PESO_PASSAGEIRO_T = 0.075 # NORMAM Padrão
        self.VELOCIDADE_VENTO_KMH = 80.0 # NORMAM Padrão

        if not self.tabela_pesos:
            logging.warning("A tabela de pesos está vazia. Os cálculos podem resultar em zero.")

    def calcular_momento_superficie_livre(self, angulo_q_graus: float) -> float:
        """
        Calcula o Momento de Superfície Livre (MSL) total para um dado ângulo
        de inclinação transversal.

        Args:
            angulo_q_graus (float): O ângulo de inclinação transversal (q) em graus.

        Returns:
            float: O MSL total em (t.m).
        """
        if not self.dados_tanques:
            return 0.0
        
        # Evita erros de divisão em ângulos como 0 e 90 graus
        if angulo_q_graus % 90 == 0:
             # Para pequenos ângulos, o MSL tende a zero. Para 90, as fórmulas são instáveis.
             # A correção real é feita no GZ via redução do GM inicial (i*p/V).
             # Esta fórmula específica é para ângulos intermediários.
            return 0.0

        q = math.radians(angulo_q_graus)
        msl_total = 0.0

        for tanque in self.dados_tanques:
            try:
                V = float(tanque['volume'])
                b = float(tanque['largura'])
                g = float(tanque['densidade'])
                L = float(tanque['comprimento'])
                h = float(tanque['altura'])
                
                if not all([V, b, g, L, h]): # Pula se algum valor for zero
                    continue

                d = V / (b * L * h)
                b_sobre_h = b / h
                tan_q = math.tan(q)
                cot_q = 1 / tan_q

                k = 0.0
                if cot_q >= b_sobre_h:
                    k = (math.sin(q) / 12) * (1 + (tan_q**2) / 2) * b_sobre_h
                else: # cot_q < b_sobre_h
                    termo1 = (math.cos(q) / 8) * (1 + tan_q / b_sobre_h)
                    termo2 = (math.cos(q) / (12 * (b**2) * (h**2))) * (1 + (cot_q**2) / 2)
                    k = termo1 - termo2

                msl_tanque = V * b * g * k * math.sqrt(d)
                msl_total += msl_tanque
                
                logging.debug(f"Tanque '{tanque['nome']}': MSL = {msl_tanque:.4f} t.m para {angulo_q_graus}°")

            except (ValueError, ZeroDivisionError) as e:
                logging.error(f"Erro ao calcular MSL para o tanque '{tanque.get('nome', 'N/A')}': {e}")
                continue

        limite_msl = 0.01 * self.deslocamento_leve
        if msl_total < limite_msl:
            logging.info(f"MSL calculado ({msl_total:.4f}) é menor que 1% do desloc. leve ({limite_msl:.4f}). Considerando MSL como 0.")
            return 0.0

        logging.info(f"MSL Total para {angulo_q_graus}°: {msl_total:.4f} t.m")
        return msl_total

    def calcular_pesos_e_momentos(self) -> Dict[str, Any]:
        """
        Calcula o peso total e os momentos longitudinal e vertical para
        cada uma das 6 condições de carregamento definidas.
        """
        print("\n--- Iniciando Cálculo de Pesos e Momentos para as Condições ---")

        logging.info("Iniciando cálculo de pesos e momentos para todas as condições.")

        for condicao in self.CONDICOES_CARREGAMENTO:
            nome_condicao = condicao["nome"]
            percentuais = condicao["percentuais"]
            logging.info(f"Processando a '{nome_condicao}'...")

            total_peso = 0.0
            total_momento_long = 0.0
            total_momento_vert = 0.0

            for item in self.tabela_pesos:
                categoria = item.get('categoria')
                # Pega o percentual da condição; se a categoria não estiver definida, usa 0
                percentual = percentuais.get(categoria, 0.0) 

                logging.debug(f"Item: {item.get('nome', 'N/A')}, Categoria: {categoria}, Percentual Aplicado: {percentual*100}%")

                peso_ajustado = float(item['peso']) * percentual
                lcg = float(item['lcg'])
                vcg = float(item['vcg'])

                momento_long = peso_ajustado * lcg
                momento_vert = peso_ajustado * vcg

                total_peso += peso_ajustado
                total_momento_long += momento_long
                total_momento_vert += momento_vert
            
            kg_condicao = 0.0
            lcg_condicao = 0.0
            if total_peso > 0:
                kg_condicao = total_momento_vert / total_peso
                lcg_condicao = total_momento_long / total_peso
            else:
                logging.warning(f"Peso total para a '{nome_condicao}' é zero. KG e LCG serão zero.")

            # Guarda os resultados totais, incluindo KG e LCG
            self.resultados_condicoes[nome_condicao] = {
                'peso_total': total_peso,
                'momento_long_total': total_momento_long,
                'momento_vert_total': total_momento_vert,
                'lcg_condicao': lcg_condicao,
                'kg_condicao': kg_condicao,
            }
            logging.info(f"Cálculo para a '{nome_condicao}' concluído.")

            # Exibe os resultados para o usuário
            print(f"\n--- {nome_condicao} ---")
            print(f"  - Peso Total: {total_peso:.3f} t")
            print(f"  - Momento Longitudinal Total: {total_momento_long:.3f} t.m")
            print(f"  - Momento Vertical Total: {total_momento_vert:.3f} t.m")
            print(f"  - LCG: {lcg_condicao:.3f} m")
            print(f"  - KG (VCG): {kg_condicao:.3f} m")

        return self.resultados_condicoes
    
    def calcular_hidrostaticas_condicoes(self):
        """
        Calcula a condição de flutuação e as características hidrostáticas para
        CADA condição de carregamento através de um processo iterativo.
        """
        logging.info("Iniciando cálculo iterativo de hidrostáticas para todas as condições.")

        for nome_condicao, dados_condicao in self.resultados_condicoes.items():
            print(f"\n--- Calculando Hidrostáticas para: {nome_condicao} ---")

            desloc_alvo = dados_condicao['peso_total']
            lcg_alvo = dados_condicao['lcg_condicao']
            kg_condicao = dados_condicao['kg_condicao']
            lpp = float(self.dados_hidrostaticos['lpp'])

            if desloc_alvo < 1e-6:
                print("   -> Deslocamento zero, pulando cálculo hidrostático.")
                continue

            # Estimativa inicial (calado médio para o deslocamento alvo, sem trim)
            # (uma função para estimar o calado a partir do deslocamento seria ideal aqui,
            # mas vamos começar com uma estimativa simples baseada no pontal)
            calado_estimado = (desloc_alvo / (self.deslocamento_leve + 100)) * (float(self.dados_hidrostaticos.get('pontal', 3.0)) * 0.8)
            calado_re_atual = calado_estimado
            calado_vante_atual = calado_estimado
            
            max_iteracoes = 100
            tolerancia = 1e-4

            for i in range(max_iteracoes):
                # Calcular as hidrostáticas para a tentativa atual
                prop_trim_iter = PropriedadesTrim(
                    calado_re=calado_re_atual, calado_vante=calado_vante_atual, lpp=lpp,
                    posicoes_balizas=self.casco.posicoes_balizas
                )
                props_iter = PropriedadesHidrostaticas(
                    interpolador_casco=self.casco, densidade=self.densidade, prop_trim=prop_trim_iter
                )
                
                desloc_calc = props_iter.deslocamento
                lcb_calc = props_iter.lcb
                mtc_calc = props_iter.mtc
                
                # Verificar convergência
                erro_desloc = (desloc_calc - desloc_alvo) / desloc_alvo
                erro_lcg = (lcb_calc - lcg_alvo) / lpp

                logging.info(f"  Iteração {i+1}: Desloc={desloc_calc:.3f}t (erro {erro_desloc:+.4%}), LCB={lcb_calc:.3f}m (erro LCG {erro_lcg:+.4%})")

                if abs(erro_desloc) < tolerancia and abs(erro_lcg) < tolerancia:
                    print(f"   -> Convergência alcançada na iteração {i+1}!")
                    
                    # Calcular resultados finais
                    calado_meio_navio = (calado_re_atual + calado_vante_atual) / 2
                    kmt = props_iter.kmt
                    gmt = kmt - kg_condicao
                    
                    # Ângulo de Alagamento
                    y_alag = float(self.ponto_alagamento.get('y', 0.0))
                    z_alag = float(self.ponto_alagamento.get('z', 0.0))
                    angulo_alagamento = 0.0
                    if y_alag > 1e-6:
                        angulo_alagamento = math.degrees(math.atan((z_alag - calado_meio_navio) / y_alag))
                        angulo_alagamento = 40 if angulo_alagamento >= 40 else angulo_alagamento

                    # Armazenar resultados
                    dados_condicao['hidrostaticos'] = {
                        'Calado Ré (m)': calado_re_atual,
                        'Calado Vante (m)': calado_vante_atual,
                        'Calado Meio-Navio (m)': calado_meio_navio,
                        'Trim (m)': calado_re_atual - calado_vante_atual,
                        'Deslocamento (t)': props_iter.deslocamento,
                        'LCB (m)': props_iter.lcb,
                        'LCF (m)': props_iter.lcf,
                        'MTC (t.m/cm)': props_iter.mtc,
                        'KMt (m)': kmt,
                        'GMt (m)': gmt,
                        'Lwl (m)': props_iter.lwl,
                        'Angulo Alagamento (°)': angulo_alagamento
                    }
                    break # Sai do loop de iterações

                # Se não convergiu, ajustar calados
                momento_trimante = desloc_alvo * (lcg_alvo - lcb_calc)
                trim_necessario = momento_trimante / (mtc_calc * 100) if mtc_calc else 0
                
                lcf_calc = props_iter.lcf
                calado_re_sem_corr = calado_re_atual - trim_necessario * ((lcf_calc / lpp) if lpp else 0.5)
                calado_vante_sem_corr = calado_vante_atual + trim_necessario * (1 - ((lcf_calc / lpp) if lpp else 0.5))
                
                tpc_iter = props_iter.tpc
                correcao_calado_medio = (desloc_alvo - desloc_calc) / (tpc_iter * 100) if tpc_iter else 0
                
                calado_re_atual = calado_re_sem_corr + correcao_calado_medio
                calado_vante_atual = calado_vante_sem_corr + correcao_calado_medio
            else: # Este 'else' pertence ao 'for', executa se o break não ocorrer
                print(f"   AVISO: O cálculo não convergiu para a '{nome_condicao}'.")

    def calcular_momento_passageiros(self, dados_condicao: Dict[str, Any], angulo_teta_graus: float) -> float:
        """Calcula o momento emborcador devido à concentração de passageiros."""
        deslocamento = dados_condicao.get('peso_total', 0.0)
        if not self.dados_passageiros or deslocamento < 1e-6:
            return 0.0

        soma_momentos_passageiros = 0
        for conves in self.dados_passageiros:
            num_passageiros = int(conves.get('num_passageiros', 0))
            yc = float(conves.get('dist_cl', 0.0))
            soma_momentos_passageiros += num_passageiros * yc

        teta_rad = math.radians(angulo_teta_graus)
        
        # A fórmula que você passou parece ser para o BRAÇO emborcador (GZ), não o momento
        # GZp = (P * soma_N_Yc * cos(teta)) / D. Vou calcular o braço.
        braco_passageiros = (self.PESO_PASSAGEIRO_T * soma_momentos_passageiros * math.cos(teta_rad)) / deslocamento

        logging.info(f"Cálculo Braço Passageiros (ângulo {angulo_teta_graus}°): "
                     f"Soma(N*Yc)={soma_momentos_passageiros:.2f}, Desloc={deslocamento:.2f}t -> GZp={braco_passageiros:.4f}m")
        
        return braco_passageiros

    def calcular_braco_guinada(self, dados_condicao: Dict[str, Any]) -> float:
        """Calcula o braço emborcador devido à força centrífuga em uma guinada."""
        hidro = dados_condicao.get('hidrostaticos')
        if not hidro:
            return 0.0
            
        deslocamento = hidro.get('Deslocamento (t)', 0.0)
        kg = dados_condicao.get('kg_condicao', 0.0)
        calado_meio_navio = hidro.get('Calado Meio-Navio (m)', 0.0)
        lwl = hidro.get('Lwl (m)', 0.0)

        if lwl < 1e-6 or deslocamento < 1e-6:
            return 0.0

        vo_ms = self.vo_knots * 0.514444 # Converte nós para m/s
        
        # A fórmula (0.02*Vo^2*D*(KG-H/2)/Lwl) / D simplifica para (0.02*Vo^2*(KG-H/2)/Lwl)
        # que é a fórmula padrão para o BRAÇO de guinada (GZg).
        braco_guinada = (0.02 * vo_ms**2 * (kg - (calado_meio_navio / 2))) / lwl
        
        logging.info(f"Cálculo Braço Guinada: Vo={self.vo_knots:.2f} nós, KG={kg:.3f}m, "
                     f"H={calado_meio_navio:.3f}m, Lwl={lwl:.2f}m -> GZg={braco_guinada:.4f}m")
        return braco_guinada

    def calcular_braco_vento(self, dados_condicao: Dict[str, Any], angulo_teta_graus: float,
                             area_velica: float, h_vento: float) -> float:
        """Calcula o braço emborcador devido à pressão do vento."""
        deslocamento = dados_condicao.get('peso_total', 0.0)
        if deslocamento < 1e-6:
            return 0.0
        
        v_vento_ms = self.VELOCIDADE_VENTO_KMH / 3.6
        teta_rad = math.radians(angulo_teta_graus)
        
        braco_vento = (5.48e-6 * area_velica * h_vento * v_vento_ms**2 * (0.25 + 0.75 * (math.cos(teta_rad)**3))) / deslocamento
        
        logging.info(f"Cálculo Braço Vento (ângulo {angulo_teta_graus}°): "
                     f"A={area_velica:.2f}m², h={h_vento:.2f}m -> GZv={braco_vento:.4f}m")
                       
        return braco_vento
    
    def gerar_curvas_estabilidade(self):
        """
        Gera a curva de estabilidade estática (GZ) final para cada condição de
        carregamento, incluindo todas as correções.
        """
        logging.info("Iniciando geração das curvas de estabilidade GZ finais.")
        
        # Define a faixa de ângulos para a análise
        angulos = list(range(0, 41, 2))

        # Instancia a calculadora de KN uma vez
        calculadora_kn = CalculadoraCurvasCruzadas(self.casco, self.df_hidrostatico, self.dados_hidrostaticos)

        for nome_condicao, dados_condicao in self.resultados_condicoes.items():
            print(f"\n--- Gerando Curva GZ para: {nome_condicao} ---")
            
            desloc_condicao = dados_condicao.get('peso_total', 0.0)
            kg_condicao = dados_condicao.get('kg_condicao', 0.0)

            if desloc_condicao < 1e-6:
                print("   -> Deslocamento zero, pulando geração da curva GZ.")
                continue

            # 1. Calcular a curva de KN para o deslocamento desta condição
            logging.info(f"Calculando KN para Deslocamento={desloc_condicao:.2f}t...")
            df_kn = calculadora_kn.calcular_curvas_kn([desloc_condicao], angulos)
            
            curva_gz_detalhada = []
            for teta in angulos:
                teta_rad = math.radians(teta)
                
                # 2. Obter o valor de KN
                # Usamos .iloc[0] para pegar a primeira (e única) linha do DataFrame
                kn_valor = df_kn[teta].iloc[0]

                # 3. Calcular a correção do KG
                kg_sin_teta = kg_condicao * math.sin(teta_rad)

                # 4. Calcular a correção de Superfície Livre (FSC)
                msl = self.calcular_momento_superficie_livre(teta)
                fsc = msl / desloc_condicao if desloc_condicao > 0 else 0

                # 5. Calcular o GZ final
                gz_final = kn_valor - kg_sin_teta - fsc
                
                curva_gz_detalhada.append({
                    'Angulo (°)': teta,
                    'KN (m)': kn_valor,
                    'KG.sen(teta) (m)': kg_sin_teta,
                    'FSC (m)': fsc,
                    'GZ (m)': gz_final
                })

            # Armazena a curva GZ completa como um DataFrame na condição
            df_gz = pd.DataFrame(curva_gz_detalhada)
            self.resultados_condicoes[nome_condicao]['curva_gz'] = df_gz
            
            # Exibe um resumo
            gz_max = df_gz['GZ (m)'].max()
            angulo_gz_max = df_gz.loc[df_gz['GZ (m)'].idxmax()]['Angulo (°)']
            print(f"   -> Curva GZ gerada. GZ máximo = {gz_max:.3f} m no ângulo de {angulo_gz_max:.1f}°")

class VerificadorCriterios:
    """
    Verifica as curvas de estabilidade com base nos critérios da NORMAM.
    """
    def __init__(self, area_navegacao: str):
        self.area_navegacao = area_navegacao
        logging.info(f"Verificador de Critérios instanciado para: {area_navegacao}")

    def _integrar_curva(self, angulos_rad: list, valores: list) -> float:
        """Calcula a área sob uma curva usando a regra dos trapézios."""
        return np.trapz(y=valores, x=angulos_rad)

    def verificar_todos(self, dados_condicao: Dict[str, Any], df_curva: pd.DataFrame) -> Dict:
        """
        Executa todas as verificações de critérios para uma dada condição.

        Args:
            dados_condicao (Dict): O dicionário da condição de carga.
            df_curva (pd.DataFrame): O DataFrame contendo as curvas GZ e de emborcamento.

        Returns:
            Dict: Um dicionário com os resultados de cada critério.
        """
        resultados = {}
        logging.info("Iniciando verificação de todos os critérios de estabilidade.")

        # --- CRITÉRIO 3: GM inicial ---
        gmo = dados_condicao.get('hidrostaticos', {}).get('GMt (m)', 0.0)
        passou_gmo = gmo >= 0.35
        logging.info(f"Critério GMo: Valor={gmo:.3f}m, Limite=0.35m, Passou={passou_gmo}")
        resultados['GM Inicial'] = {'valor': f"{gmo:.3f} m", 'esperado': ">= 0.35 m", 'passou': passou_gmo}

        # --- CRITÉRIO 5: GZ Máximo ---
        gz_max = df_curva['GZ (m)'].max()
        passou_gz_max = gz_max >= 0.15
        logging.info(f"Critério GZ Máximo: Valor={gz_max:.3f}m, Limite=0.15m, Passou={passou_gz_max}")
        resultados['GZ Máximo'] = {'valor': f"{gz_max:.3f} m", 'esperado': ">= 0.15 m", 'passou': passou_gz_max}

        # --- CRITÉRIO 4: Ângulo de Alagamento ---
        theta_f = dados_condicao.get('hidrostaticos', {}).get('Angulo Alagamento (°)', 0.0)
        limite_theta_f = 25.0 if "Área 1" in self.area_navegacao else 30.0
        passou_theta_f = theta_f >= limite_theta_f
        logging.info(f"Critério Ângulo Alagamento: Valor={theta_f:.2f}°, Limite={limite_theta_f}°, Passou={passou_theta_f}")
        resultados['Ângulo de Alagamento'] = {'valor': f"{theta_f:.2f}°", 'esperado': f">= {limite_theta_f}°", 'passou': passou_theta_f}


        # --- CRITÉRIOS 1 E 2 (requerem braço de emborcamento) ---
        if 'GZ Emborcador (m)' in df_curva.columns:
            gz_heeling = df_curva['GZ Emborcador (m)']
            gz_righting = df_curva['GZ (m)']
            
            # Encontra o ângulo de equilíbrio estático (theta_1)
            intersecao_idx = np.where(np.diff(np.sign(gz_righting - gz_heeling)))[0]
            theta_1 = 0.0
            if len(intersecao_idx) > 0:
                # Interpola para um valor mais preciso do ângulo de interseção
                idx = intersecao_idx[0]
                x1, x2 = df_curva['Angulo (°)'].iloc[idx], df_curva['Angulo (°)'].iloc[idx+1]
                y1 = gz_righting.iloc[idx] - gz_heeling.iloc[idx]
                y2 = gz_righting.iloc[idx+1] - gz_heeling.iloc[idx+1]
                theta_1 = x1 - y1 * (x2 - x1) / (y2 - y1)

            # CRITÉRIO 1: Ângulo de Equilíbrio
            limite_theta_1 = 15.0 if "Área 1" in self.area_navegacao else 12.0
            passou_theta_1 = theta_1 <= limite_theta_1
            resultados['Ângulo de Equilíbrio'] = {'valor': f"{theta_1:.2f}°", 'esperado': f"<= {limite_theta_1}°", 'passou': passou_theta_1}
            
            # 1. Usamos np.radians para converter a coluna inteira de uma vez
            df_curva['Angulo (rad)'] = np.radians(df_curva['Angulo (°)'])
            
            # 2. Filtramos o DataFrame usando a nova coluna
            theta_f = dados_condicao.get('hidrostaticos', {}).get('Angulo Alagamento (°)', 0.0)
            limite_area_rad = math.radians(min(40.0, theta_f))
            df_area = df_curva[df_curva['Angulo (rad)'] <= limite_area_rad].copy()

            # Se não houver dados para calcular a área (ex: alagamento em ângulo baixo), evitamos erros
            if len(df_area) < 2:
                area_a_residual = 0.0
                area_b_heeling = 0.0
            else:
                # 3. Usamos a coluna em radianos diretamente para os cálculos
                angulos_rad_area = df_area['Angulo (rad)']
                
                # Filtra os dados para antes e depois de theta_1 para as integrais
                area_b_df = df_area[df_area['Angulo (rad)'] <= math.radians(theta_1)]
                area_a_df = df_area[df_area['Angulo (rad)'] >= math.radians(theta_1)]
                
                area_b_heeling = self._integrar_curva(area_b_df['Angulo (rad)'],
                                                      area_b_df['GZ Emborcador (m)'] - area_b_df['GZ (m)'])
                
                area_a_residual = self._integrar_curva(area_a_df['Angulo (rad)'],
                                                       (area_a_df['GZ (m)'] - area_a_df['GZ Emborcador (m)']))
            
            fator_area = 1.0 if "Área 1" in self.area_navegacao else 1.2
            passou_area = area_a_residual >= (fator_area * area_b_heeling)
            resultados['Área Residual'] = {'valor': f"{area_a_residual:.4f} m.rad", 'esperado': f">= {fator_area * area_b_heeling:.4f} m.rad", 'passou': passou_area}
            
        return resultados

if __name__ == '__main__':
    import os
    import pandas as pd

    from ..io.file_handler import FileHandler
    from ..core.ch import CalculadoraHidrostatica
    from ..utils.list_utils import gerar_lista_de_calados
    from ..ui.plotting import plotar_curva_estabilidade

    print("="*50)
    print("--- EXECUTANDO eed.py EM MODO DE TESTE ---")
    print("="*50)

    # --- 1. PREPARAR DADOS DE TESTE (CASCO E DADOS HIDROSTÁTICOS) ---
    try:
        diretorio_raiz_projeto = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        caminho_csv = os.path.join(diretorio_raiz_projeto, "data", "exemplos_tabelas_cotas", "TABELA DE COTAS.csv")
        tabela_bruta = pd.read_csv(caminho_csv)
        
        from ..io.file_handler import FileHandler
        manipulador_arquivos_teste = FileHandler()
        densidade_teste = 1.025
        dados_hidro_teste = {'lpp': 19.713, 'pontal': 3.0, 'boca': 6.0, 'referencial': 'Perpendicular de ré (AP)', 'densidade': densidade_teste}
        
        tabela_processada = manipulador_arquivos_teste.processar_dados_balizas(
            tabela_bruta,
            lpp=dados_hidro_teste['lpp'],
            referencial_saida=dados_hidro_teste['referencial']
        )
        casco_teste = InterpoladorCasco(tabela_processada, metodo_interp='Linear')
        
        print(f"-> Casco de teste carregado de '{caminho_csv}' com sucesso.")

        print("-> Gerando DataFrame hidrostático de base para o teste...")
        calados_teste = gerar_lista_de_calados({'metodo': 'numero', 'min': 0, 'max': 3.0, 'num': 5})
        calculadora_hidro_teste = CalculadoraHidrostatica(casco_teste, densidade=densidade_teste)
        df_hidrostatico_teste = calculadora_hidro_teste.calcular_curvas(calados_teste)
        print("-> DataFrame hidrostático gerado.")
    
    except FileNotFoundError:
        print("\nERRO DE TESTE: O arquivo 'TABELA DE COTAS.csv' não foi encontrado.")
        casco_teste = None
    except Exception as e:
        print(f"\nERRO DE TESTE ao carregar o casco: {e}")
        casco_teste = None

    if casco_teste:
        # --- 2. PREPARAR DADOS DE EXEMPLO (MOCK) ---
        mock_deslocamento_leve = 90.8445
        mock_lcg_leve = 9.6464
        mock_kg_leve = 2.3496
        mock_ponto_alagamento = {'y': 0.58, 'z': 3.9}
        mock_velocidade = 10.0 # nós

        print(f"Usando Deslocamento Leve de Teste: {mock_deslocamento_leve:.4f} t")

        mock_dados_estabilidade = {
            "tabela_pesos": [
                {'nome': 'Embarcação Leve', 'peso': str(mock_deslocamento_leve), 'lcg': str(mock_lcg_leve), 'vcg': str(mock_kg_leve), 'categoria': 'EMBARCAÇÃO LEVE'},
                {'nome': 'TQ. DE ÁGUA DOCE (CASCO)', 'peso': '12.8', 'lcg': '17.956', 'vcg': '1.5', 'categoria': 'GENEROS E ÓLEOS'},
                {'nome': 'TQ. DE ÓLEO DIESEL RÉ', 'peso': '25.81', 'lcg': '2.0565', 'vcg': '2.27', 'categoria': 'GENEROS E ÓLEOS'},
                {'nome': 'TQ. DE ÓLEO DIESEL 03 BE', 'peso': '7.85', 'lcg': '11.2365', 'vcg': '2.1', 'categoria': 'GENEROS E ÓLEOS'},
                {'nome': 'TQ. DE ÓLEO DIESEL 03 BB', 'peso': '7.85', 'lcg': '11.2365', 'vcg': '2.1', 'categoria': 'GENEROS E ÓLEOS'},
                {'nome': 'TQ. ÓLEO DE CONSUMO', 'peso': '1.573', 'lcg': '15.0035', 'vcg': '2.5', 'categoria': 'GENEROS E ÓLEOS'},
                {'nome': 'PORÃO DE CARGA (GELO)', 'peso': '9.569', 'lcg': '6.8265', 'vcg': '2', 'categoria': 'GENEROS E ÓLEOS'},
                {'nome': 'TQ. DE ÁGUA DOCE (TIJUPÁ)', 'peso': '0.2', 'lcg': '11.5565', 'vcg': '6.4', 'categoria': 'GENEROS E ÓLEOS'},
                {'nome': 'TRIPULAÇÃO + PERTENCES (4 TRIP. + 2 E.ROLL)', 'peso': '0.6', 'lcg': '14.8565', 'vcg': '4.3', 'categoria': 'TRIPULAÇÃO'},
            ],
            "ponto_alagamento": mock_ponto_alagamento,
            "dados_tanques": [
                {'nome': 'TQ. DE ÁGUA DOCE (CASCO)', 'volume': '12.25', 'comprimento': '5.885', 'largura': '5.896', 'altura': ' 4.322', 'densidade': '1.0'},
                {'nome': 'TQ. DE ÓLEO DIESEL RÉ', 'volume': '38.98', 'comprimento': '4.458', 'largura': '2.75', 'altura': '5.738', 'densidade': '0.85'},
                {'nome': 'TQ. DE ÓLEO DIESEL 03 BE', 'volume': '8.94', 'comprimento': '2.63', 'largura': '3.3', 'altura': '1.464', 'densidade': '0.85'},
                {'nome': 'TQ. DE ÓLEO DIESEL 03 BB', 'volume': '8.94', 'comprimento': '2.63', 'largura': '3.3', 'altura': '1.464', 'densidade': '0.85'},
                {'nome': 'TQ. ÓLEO DE CONSUMO', 'volume': '1.088', 'comprimento': '0.5', 'largura': '1.5', 'altura': '1.45', 'densidade': '0.85'},
                {'nome': 'TQ. DE ÁGUA DOCE (TIJUPÁ)', 'volume': '0.2', 'comprimento': '0.585', 'largura': '0.585', 'altura': '0.585', 'densidade': '1.0'},
            ],
            "dados_passageiros": [],
            "velocidade_operacao": str(mock_velocidade)
        }

        # --- 3. EXECUTAR OS CÁLCULOS ---
        calculadora_teste = CalculadoraEED(
            dados_estabilidade=mock_dados_estabilidade,
            deslocamento_leve=mock_deslocamento_leve,
            casco=casco_teste,
            dados_hidrostaticos=dados_hidro_teste,
            df_hidrostatico=df_hidrostatico_teste,
            densidade=densidade_teste
        )
        
        calculadora_teste.calcular_pesos_e_momentos()
        calculadora_teste.calcular_hidrostaticas_condicoes()
        calculadora_teste.gerar_curvas_estabilidade()

        # --- 4. EXIBIR OS RESULTADOS FINAIS DO TESTE (CURVA GZ) ---
        print("\n" + "="*60)
        print("--- VERIFICAÇÃO DE CRITÉRIOS NO MODO DE TESTE ---")
        
        # Simula a escolha do usuário para a área de navegação
        area_nav_teste = "Área 2"
        verificador_teste = VerificadorCriterios(area_nav_teste)
        print(f"-> Testando para: {area_nav_teste}")
        
        # Simula os dados de vento que seriam inseridos pelo usuário
        mock_area_velica = 55.0
        mock_h_vento = 3.2
        
        # Pega a primeira condição de carga como exemplo para exibir e plotar
        nome_primeira_cond = next(iter(calculadora_teste.resultados_condicoes.keys()))
        primeira_condicao = calculadora_teste.resultados_condicoes[nome_primeira_cond]
        
        if 'hidrostaticos' in primeira_condicao and 'curva_gz' in primeira_condicao:
            print(f"\n--- ANÁLISE DA CONDIÇÃO: {nome_primeira_cond} ---")
            
            df_gz_teste = primeira_condicao['curva_gz'].copy()
            
            # Adiciona os braços de emborcamento ao DataFrame para o teste
            bracos_emborcadores = []
            for angulo in df_gz_teste['Angulo (°)']:
                braco_p = calculadora_teste.calcular_momento_passageiros(primeira_condicao, angulo)
                braco_g = calculadora_teste.calcular_braco_guinada(primeira_condicao)
                braco_v = calculadora_teste.calcular_braco_vento(primeira_condicao, angulo, mock_area_velica, mock_h_vento)
                bracos_emborcadores.append(braco_g + braco_v + braco_p)
            df_gz_teste['GZ Emborcador (m)'] = bracos_emborcadores
            
            # Chama o verificador
            resultados_criterios = verificador_teste.verificar_todos(primeira_condicao, df_gz_teste)
            
            # Exibe os resultados da verificação
            print("\n  -> Resultado da Verificação:")
            for criterio, resultado in resultados_criterios.items():
                status = "PASSOU" if resultado['passou'] else "FALHOU"
                print(f"    - {criterio:<22} | Status: {status}")

            # --- CHAMADA PARA A PLOTAGEM ---
            print("\n  -> Gerando gráfico de estabilidade para a condição de teste...")
            
            # Define um caminho para salvar o gráfico de teste
            caminho_salvar_teste = os.path.join(diretorio_raiz_projeto, "data", "projetos_salvos", "_teste", "grafico_teste.png")
            
            plotar_curva_estabilidade(
                df_curva=df_gz_teste,
                resultados_criterios=resultados_criterios,
                dados_condicao=primeira_condicao,
                nome_condicao=nome_primeira_cond,
                caminho_salvar=caminho_salvar_teste
            )
        print("="*60)

    print("\n--- TESTE CONCLUÍDO ---")