import pandas as pd
import numpy as np
from typing import Dict, Any

class CalculadoraRPI:
    """
    Encapsula todos os cálculos relacionados com o Relatório da Prova de Inclinação.
    """
    def __init__(self, dados_rpi: Dict[str, Any], dados_hidrostaticos: Dict[str, Any]):
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
        
        # Resultados que serão calculados
        self.calados_nas_marcas: Dict[str, float] = {}
        self.calados_nas_perpendiculares: Dict[str, float] = {}

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
            HMM = float(dados_flutuacao['pontal_meio']) - bl_meio
            HMV = float(dados_flutuacao['pontal_vante']) - bl_vante
        
        else: # "Leitura direta dos calados"
            print("-> A usar calados lidos diretamente...")
            # Assume-se que a banda é desprezível, então a leitura de um bordo é a média.
            HMR = (float(dados_flutuacao['bb_re']) + float(dados_flutuacao['be_re'])) / 2
            HMM = (float(dados_flutuacao['bb_meio']) + float(dados_flutuacao['be_meio'])) / 2
            HMV = (float(dados_flutuacao['bb_vante']) + float(dados_flutuacao['be_vante'])) / 2
        
        self.calados_nas_marcas = {"re": HMR, "meio": HMM, "vante": HMV}
        print(f"Calados médios nas marcas: Ré={HMR:.4f}m, Meio={HMM:.4f}m, Vante={HMV:.4f}m")

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
        HMN = HMN + dHMN

        # 6. Correção para a Perpendicular de Vante (HPV)
        dHPV = LV * tan_theta
        HPV = HMV - dHPV # NOTA: Usado sinal negativo. Se HMR > HMV (trim pela popa), o calado na PV é menor.

        self.calados_nas_perpendiculares = {"re": HPR, "meio": HMN, "vante": HPV}
        print(f"Calados corrigidos nas perpendiculares: PR={HPR:.4f}m, MN={HMN:.4f}m, PV={HPV:.4f}m")