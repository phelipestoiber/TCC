# src/ui/menu.py

import questionary
import os
from typing import Dict, Any, Optional, List
import pandas as pd
from math import floor

class Menu:
    """
    Gerencia a interface do usuário no terminal para a entrada de dados.
    """

    def _validar_float_positivo(self, text: str) -> bool:
        """Validador para garantir que a entrada seja um float positivo ou zero."""
        if not text:
            return False
        try:
            value = float(text)
            return value >= 0
        except ValueError:
            return False
        
    def _validar_float_qualquer(self, text: str) -> bool:
        """Validador para garantir que a entrada seja um float qualquer (pos, neg ou zero)."""
        if not text: return False
        try:
            float(text)
            return True
        except ValueError:
            return False
        
    def _validar_int_positivo(self, text: str) -> bool:
        """Validador para garantir que a entrada seja um inteiro positivo."""
        if not text: return False
        try:
            value = int(text)
            return value > 0
        except ValueError:
            return False

    def _validar_int_min_dois(self, text: str) -> bool:
        """Validador para garantir que a entrada seja um inteiro >= 2."""
        if not text:
            return False
        try:
            value = int(text)
            return value >= 2
        except ValueError:
            return False

    def _validar_listas(self, text: str) -> bool:
        """Validador genérico para uma string de números separados por ';'."""
        if not text:
            return False
        try:
            [float(c.strip()) for c in text.split(';')]
            return True
        except ValueError:
            return False
        
    def _validar_lista_com_5_numeros(self, text: str) -> bool:
        """Validador para uma string que deve conter exatamente 5 números."""
        if not text: return False
        try:
            partes = [p.strip() for p in text.split(';') if p.strip()]
            if len(partes) != 5:
                # Retorna uma mensagem de erro específica para o questionary
                raise questionary.ValidationError(
                    message="Por favor, insira exatamente 5 valores separados por ';'."
                )
            # Tenta converter todos para float para garantir que são números
            for parte in partes:
                float(parte)
            return True
        except (ValueError, IndexError):
            return False
        
    def deseja_prosseguir(self, proximo_passo: str) -> bool:
        """
        Apresenta uma pergunta de confirmação genérica ao utilizador.

        Args:
            proximo_passo (str): O nome do próximo módulo/cálculo.

        Returns:
            bool: True se o utilizador confirmar, False caso contrário.
        """
        return questionary.confirm(
            f"Deseja prosseguir para o {proximo_passo}?",
            default=True
        ).ask()

    def obter_dados_hidrostaticos(self) -> Dict[str, Any]:
        """
        Coleta todos os dados de entrada necessários do usuário de forma interativa.
        """
        # print("--- TCC Naval - Entrada de Dados ---")
        print("Por favor, forneça as informações a seguir:\n")

        # 1. Obtem o nome da embarcação/projeto e o caminho do arquivo CSV
        nome_projeto = questionary.text("Qual o nome do projeto ou embarcação?", default="teste").ask()
        if not nome_projeto: print("Operação cancelada."); exit()

        caminho_arquivo = ""
        while not caminho_arquivo:
            path = questionary.path(
                "Qual o caminho para o arquivo da tabela de cotas (.csv)?",
                validate=lambda p: os.path.exists(p) and p.lower().endswith('.csv'),
                default="C:\\Users\\afmn\\Desktop\\TCC\\data\\exemplos_tabelas_cotas\\TABELA DE COTAS.csv"
            ).ask()

            if path:
                confirmado = questionary.confirm(f"O caminho '{path}' está correto?").ask()
                if confirmado:
                    caminho_arquivo = path
            else:
                print("Operação cancelada.")
                exit()
        
        # 2. Obter características da embarcação e densidade
        dados_embarcacao = questionary.form(
            lpp=questionary.text("Comprimento entre perpendiculares (Lpp) em metros:", validate=self._validar_float_positivo, default="19.713"),
            boca=questionary.text("Boca moldada em metros:", validate=self._validar_float_positivo, default="6"),
            pontal=questionary.text("Pontal moldado em metros:", validate=self._validar_float_positivo, default="3"),
            densidade=questionary.text("Densidade da água (ex: 1.025):", default="1.025", validate=self._validar_float_positivo)
        ).ask()

        if not dados_embarcacao:
            print("Operação cancelada."); exit()

        # 3. Obter configurações de cálculo
        referencial = questionary.select(
            "Qual o referencial para os resultados dos cálculos?",
            choices=["Perpendicular de ré (AP)", "Perpendicular de meio-navio (MS)"]
        ).ask()

        metodo_interp = questionary.select(
            "Qual método de interpolação deseja usar nos cálculos?",
            choices=["Linear", "PCHIP (Cúbica, preserva a monotonia)"],
            instruction="Linear é mais rápido. PCHIP gera curvas mais suaves."
        ).ask()
        
        if not referencial or not metodo_interp:
            print("Operação cancelada."); exit()
            
        # 4. Obter a forma de definir os calados
        metodo_calado = questionary.select(
            "Como você deseja definir a faixa de calados para os cálculos?",
            choices=["Definir calado mínimo, máximo e o número de calados", "Definir calado mínimo, máximo e o passo", "Fornecer uma lista de calados"]
        ).ask()

        if not metodo_calado:
            print("Operação cancelada."); exit()

        dados_calado = {}
        if "lista" in metodo_calado:
            calados_str = questionary.text("Digite os calados separados por ponto e vírgula (ex: 0.5; 1.0; 1.5):", validate=self._validar_listas).ask()
            dados_calado = {"metodo": "lista", "valores": calados_str}
        
        else: # Opções de min/max
            # Utiliza um form para pegar min e max juntos, o que é mais robusto
            calados_min_max = questionary.form(
                calado_min=questionary.text("Calado mínimo:", validate=lambda val: self._validar_float_positivo(val) and float(val) < float(dados_embarcacao['pontal']), default="0"),
                calado_max=questionary.text("Calado máximo:", validate=lambda val: self._validar_float_positivo(val) and float(val) <= float(dados_embarcacao['pontal']), default="3")
            ).ask()

            if not calados_min_max:
                print("Operação cancelada."); exit()
            
            calado_min = float(calados_min_max['calado_min'])
            calado_max = float(calados_min_max['calado_max'])

            if calado_min >= calado_max:
                print("\nErro: O calado mínimo deve ser menor que o calado máximo.")
                exit()

            if "número" in metodo_calado:
                num_calados = int(questionary.text("Número de calados:", validate=self._validar_int_min_dois, default="15").ask())
                dados_calado = {"metodo": "numero", "min": calado_min, "max": calado_max, "num": num_calados}
            
            else: # Opção de passo
                passo = float(questionary.text("Passo entre os calados:", validate=lambda val: self._validar_float_positivo(val) and float(val) <= (calado_max - calado_min)).ask())
                dados_calado = {"metodo": "passo", "min": calado_min, "max": calado_max, "passo": passo}

        # 5. Unir todos os dados em um único dicionário
        dados_finais = {
            "nome_projeto": nome_projeto,
            "caminho_arquivo": caminho_arquivo,
            **dados_embarcacao,
            "metodo_interp": metodo_interp,
            "referencial": referencial,
            "calados": dados_calado,
        }
        
        return dados_finais
    
    def obter_dados_curvas_cruzadas(self, df_hidrostatico: pd.DataFrame) -> Dict[str, Any]:
        """Coleta os dados de entrada para o cálculo das curvas cruzadas."""
        print("\n--- Entrada de Dados para Curvas Cruzadas ---")
        max_desloc = df_hidrostatico['Desloc. (t)'].max()
        print(f"INFO: O deslocamento máximo da hidrostática é {max_desloc:.4f} t.")

        # 1. Obter a forma de definir os deslocamentos
        metodo_deslocamento = questionary.select(
            "Como você deseja definir a faixa de deslocamentos para os cálculos?",
            choices=["Definir deslocamento mínimo, máximo e o número de deslocamentos", "Definir deslocamento mínimo, máximo e o passo", "Fornecer uma lista de deslocamentos"]
        ).ask()

        if not metodo_deslocamento:
            print("Operação cancelada."); exit()

        dados_deslocamento = {}
        if "lista" in metodo_deslocamento:
            deslocamentos_str = questionary.text("Digite os deslocamentos separados por ponto e vírgula (ex: 0.5; 1.0; 1.5):", validate=self._validar_listas).ask()
            dados_deslocamento = {"metodo": "lista", "valores": deslocamentos_str}
        
        else: # Opções de min/max
            # Utiliza um form para pegar min e max juntos, o que é mais robusto
            deslocamentos_min_max = questionary.form(
                deslocamento_min=questionary.text("Deslocamento mínimo:", validate=lambda val: self._validar_float_positivo(val), default="0"),
                deslocamento_max=questionary.text("Deslocamento máximo:", validate=lambda val: self._validar_float_positivo(val) and float(val) <= float(max_desloc), default=f"{floor(max_desloc)}")
            ).ask()

            if not deslocamentos_min_max:
                print("Operação cancelada."); exit()
            
            deslocamento_min = float(deslocamentos_min_max['deslocamento_min'])
            deslocamento_max = float(deslocamentos_min_max['deslocamento_max'])

            if deslocamento_min >= deslocamento_max:
                print("\nErro: O deslocamento mínimo deve ser menor que o deslocamento máximo.")
                exit()

            if "número" in metodo_deslocamento:
                num_deslocamentos = int(questionary.text("Número de deslocamentos:", validate=self._validar_int_min_dois, default="15").ask())
                dados_deslocamento = {"metodo": "numero", "min": deslocamento_min, "max": deslocamento_max, "num": num_deslocamentos}
            
            else: # Opção de passo
                passo = float(questionary.text("Passo entre os deslocamentos:", validate=lambda val: self._validar_float_positivo(val) and float(val) <= (deslocamento_max - deslocamento_min)).ask())
                dados_deslocamento = {"metodo": "passo", "min": deslocamento_min, "max": deslocamento_max, "passo": passo}

        # 2. Obter a forma de definir os ângulos
        metodo_angulo = questionary.select(
            "Como você deseja definir a faixa de angulos para os cálculos?",
            choices=["Definir ângulo mínimo, máximo e o número de angulos", "Definir ângulo mínimo, máximo e o passo", "Fornecer uma lista de ângulos"]
        ).ask()

        if not metodo_angulo:
            print("Operação cancelada."); exit()

        dados_angulo = {}
        if "lista" in metodo_angulo:
            angulos_str = questionary.text("Digite os ângulos separados por ponto e vírgula (ex: 0.5; 1.0; 1.5):", validate=self._validar_listas).ask()
            dados_angulo = {"metodo": "lista", "valores": angulos_str}
        
        else: # Opções de min/max
            # Utiliza um form para pegar min e max juntos, o que é mais robusto
            angulos_min_max = questionary.form(
                angulo_min=questionary.text("Ângulo mínimo:", validate=lambda val: self._validar_float_positivo(val), default="0"),
                angulo_max=questionary.text("Ângulo máximo:", validate=lambda val: self._validar_float_positivo(val) and float(val) <= float("90"), default="60")
            ).ask()

            if not angulos_min_max:
                print("Operação cancelada."); exit()
            
            angulo_min = float(angulos_min_max['angulo_min'])
            angulo_max = float(angulos_min_max['angulo_max'])

            if angulo_min >= angulo_max:
                print("\nErro: O ângulo mínimo deve ser menor que o angulo máximo.")
                exit()

            if "número" in metodo_angulo:
                num_angulos = int(questionary.text("Número de ângulos:", validate=self._validar_int_min_dois, default="13").ask())
                dados_angulo = {"metodo": "numero", "min": angulo_min, "max": angulo_max, "num": num_angulos}
            
            else: # Opção de passo
                passo = float(questionary.text("Passo entre os ângulos:", validate=lambda val: self._validar_float_positivo(val) and float(val) <= (angulo_max - angulo_min)).ask())
                dados_angulo = {"metodo": "passo", "min": angulo_min, "max": angulo_max, "passo": passo}
            
        dados_finais = {"deslocamentos": dados_deslocamento, "angulos": dados_angulo}
            
        return dados_finais
    
    def obter_dados_rpi(self) -> Dict[str, Any]:
        """Recolhe os dados de entrada iniciais para o Relatório de Prova de Inclinação."""
        print("\n--- Entrada de Dados para a Prova de Inclinação ---")

        # Metodo de medição dos ângulos de inclinação
        metodo_inclinacao = questionary.select(
            "Qual método foi usado para medir os ângulos de inclinação?",
            choices=["Pêndulos", "Tubos em U"]
        ).ask()
        if not metodo_inclinacao: print("Operação cancelada."); exit()

        # Tipo de pesos movimentados
        tipo_pesos = questionary.select(
            "Qual tipo de peso foi movimentado durante a prova?",
            choices=["Pesos sólidos", "Pesos líquidos (lastro)"]
        ).ask()
        if not tipo_pesos: print("Operação cancelada."); exit()

        # Tipo de medição da condição de flutuação
        metodo_flutuacao = questionary.select(
            "Como foi determinada a condição de flutuação da embarcação?",
            choices=[
                "Leitura direta dos calados",
                "Medição das bordas livres"
            ]
        ).ask()
        if not metodo_flutuacao: print("Operação cancelada."); exit()

        dados_flutuacao = {"metodo": metodo_flutuacao}
        
        # Condicional para leitura de calados
        if "calados" in metodo_flutuacao:
            print("\nPor favor, insira os calados lidos.")
            print("\nNOTA: Assume-se LR e LM positivos a vante das perpendiculares e LV positivo a ré.")
            calados_lidos = questionary.form(
                # Posições longitudinais das marcas de calado
                lr=questionary.text("Distância da marca de calado de ré à perpendicular de ré (LR) [m]:", validate=self._validar_float_qualquer, default="0"),
                lm=questionary.text("Distância da marca de calado de Meio-Navio à perpendicular de Meio-Navio (LM) [m]:", validate=self._validar_float_qualquer, default="0"),
                lv=questionary.text("Distância da marca de calado de vante à perpendicular de vante (LV) [m]:", validate=self._validar_float_qualquer, default="0"),
                # Leituras dos calados
                bb_re=questionary.text("Calado a bombordo, a Ré [m]:", validate=self._validar_float_positivo, default="1.875"),
                bb_meio=questionary.text("Calado a bombordo, a Meio-Navio [m]:", validate=self._validar_float_positivo, default="1.835"),
                bb_vante=questionary.text("Calado a bombordo, a Vante [m]:", validate=self._validar_float_positivo, default="1.775"),
                be_re=questionary.text("Calado a boreste, a Ré [m]:", validate=self._validar_float_positivo, default="1.875"),
                be_meio=questionary.text("Calado a boreste, a Meio-Navio [m]:", validate=self._validar_float_positivo, default="1.835"),
                be_vante=questionary.text("Calado a boreste, a Vante [m]:", validate=self._validar_float_positivo, default="1.775"),
            ).ask()
            if not calados_lidos: print("Operação cancelada."); exit()
            dados_flutuacao.update(calados_lidos)

        # Condicional para medição de bordas livres
        else: # "bordas livres"
            print("\nPor favor, insira as informações sobre a medição das bordas livres.")
            dados_bordas_livres = questionary.form(
                # Posições longitudinais das marcas de calado
                lr=questionary.text("Posição longitudinal no local da medição a Ré (LR) [m]:", validate=self._validar_float_qualquer, default="13.039"),
                lm=questionary.text("Posição longitudinal no local da medição a Meio-Navio (LM) [m]:", validate=self._validar_float_qualquer, default="3.815"),
                lv=questionary.text("Posição longitudinal no local da medição a Vante (LV) [m]:", validate=self._validar_float_qualquer, default="16.433"),
                # Pontais nos locais de medição
                pontal_re=questionary.text("Pontal moldado no local da medição a Ré [m]:", validate=self._validar_float_positivo, default="3.125"),
                pontal_meio=questionary.text("Pontal moldado no local da medição a Meio-Navio [m]:", validate=self._validar_float_positivo, default="3"),
                pontal_vante=questionary.text("Pontal moldado no local da medição a Vante [m]:", validate=self._validar_float_positivo, default="3.079"),
                # Leituras das bordas livres
                bl_bb_re=questionary.text("Borda Livre a Bombordo, a Ré [m]:", validate=self._validar_float_positivo, default="1.918"),
                bl_be_re=questionary.text("Borda Livre a Boreste, a Ré [m]:", validate=self._validar_float_positivo, default="1.918"),
                bl_bb_meio=questionary.text("Borda Livre a Bombordo, a Meio-Navio [m]:", validate=self._validar_float_positivo, default="1.934"),
                bl_be_meio=questionary.text("Borda Livre a Boreste, a Meio-Navio [m]:", validate=self._validar_float_positivo, default="1.934"),
                bl_bb_vante=questionary.text("Borda Livre a Bombordo, a Vante [m]:", validate=self._validar_float_positivo, default="2.221"),
                bl_be_vante=questionary.text("Borda Livre a Boreste, a Vante [m]:", validate=self._validar_float_positivo, default="2.221"),
            ).ask()
            if not dados_bordas_livres: print("Operação cancelada."); exit()
            dados_flutuacao.update(dados_bordas_livres)

        # Densidades medidas no local
        print("\nPor favor, insira as densidades da água [t/m³] medidas no local.")
        densidades_medidas = questionary.form(
            re=questionary.text("Densidade a Ré:", validate=self._validar_float_positivo, default="1.025"),
            meio=questionary.text("Densidade a Meio-Navio:", validate=self._validar_float_positivo, default="1.025"),
            vante=questionary.text("Densidade a Vante:", validate=self._validar_float_positivo, default="1.025"),
        ).ask()
        if not densidades_medidas: print("Operação cancelada."); exit()

        # Dados dos tanques (se houver)
        lista_tanques = []
        if questionary.confirm("\nForam sondados tanques com líquidos a bordo?", default=False).ask():
            num_tanques_str = questionary.text(
                "Quantos tanques foram sondados?",
                validate=self._validar_int_positivo
            ).ask()
            num_tanques = int(num_tanques_str) if num_tanques_str else 0

            for i in range(num_tanques):
                print(f"\n--- Dados para o Tanque nº {i+1}/{num_tanques} ---")
                dados_tanque = questionary.form(
                    nome=questionary.text("Nome do Tanque:"),
                    sondagem=questionary.text("Altura de Sondagem/Ulage [m]:", validate=self._validar_float_positivo),
                    volume=questionary.text("Volume do líquido no tanque [m³]:", validate=self._validar_float_positivo),
                    pe=questionary.text("Peso Específico do líquido [t/m³]:", validate=self._validar_float_positivo),
                    lcg=questionary.text("Posição Longitudinal do CG do tanque (LCG) [m]:", validate=self._validar_float_qualquer),
                    vcg=questionary.text("Posição Vertical do CG do tanque (VCG) [m]:", validate=self._validar_float_positivo),
                    mls=questionary.text("Momento de Superfície Livre do tanque [t.m]:", validate=self._validar_float_positivo),
                ).ask()
                if not dados_tanque: print("Operação cancelada."); exit()
                lista_tanques.append(dados_tanque)

        # Itens a deduzir (se houver)
        #  1. Pessoas a bordo (sempre)
        print("\n--- Itens a Deduzir (Automático) ---")
        dados_pessoas = questionary.form(
            peso=questionary.text("Peso total das pessoas a bordo [t]:", validate=self._validar_float_positivo, default="0.32"),
            lcg=questionary.text("LCG médio das pessoas [m]:", validate=self._validar_float_qualquer, default="6.8565"),
            vcg=questionary.text("VCG médio das pessoas [m]:", validate=self._validar_float_positivo, default="4.3"),
        ).ask()
        if not dados_pessoas: print("Operação cancelada."); exit()

        # Perguntar sobre os 4 pesos da prova (apenas se forem sólidos)
        lista_pesos_prova = []
        if "sólidos" in tipo_pesos:
            print("\n--- Dados para os 4 Pesos da Prova de Inclinação ---")
            # Loop para recolher os dados de cada um dos 4 pesos
            for i in range(4):
                print(f"--- Dados para o Peso da Prova nº {i+1}/4 ---")
                dados_peso_individual = questionary.form(
                    peso=questionary.text("Peso [t]:", validate=self._validar_float_positivo, default="0.2"),
                    lcg=questionary.text("Posição Longitudinal (LCG) [m]:", validate=self._validar_float_qualquer),
                    vcg=questionary.text("Posição Vertical (VCG) [m]:", validate=self._validar_float_positivo),
                    tcg=questionary.text("Posição Transversal (TCG) [m]:", validate=self._validar_float_qualquer),
                ).ask()
                if not dados_peso_individual: print("Operação cancelada."); exit()
                lista_pesos_prova.append(dados_peso_individual)

        # Perguntar sobre OUTROS itens a deduzir
        # 3. Outros itens a deduzir (se houver)
        outros_itens_a_deduzir = []
        if questionary.confirm("\nHá OUTROS pesos a serem deduzidos (ex: lixo, equipamentos extras)?", default=False).ask():
            outros_itens_a_deduzir = self._obter_lista_de_itens("Outros Itens a Deduzir")

        # --- LÓGICA DE CONSTRUÇÃO DA LISTA FINAL DE DEDUÇÕES ---
        itens_a_deduzir = []
        # Adiciona o item "Pessoas a bordo" à lista
        itens_a_deduzir.append({
            "nome": "Pessoas a bordo",
            **dados_pessoas
        })
        # Adiciona cada um dos 4 pesos da prova à lista
        if lista_pesos_prova:
            for i, peso_info in enumerate(lista_pesos_prova):
                itens_a_deduzir.append({
                    "nome": f"Peso da prova de inclinação nº {i+1}",
                    **peso_info
                })
        # Adiciona os outros itens que o utilizador inseriu manualmente
        itens_a_deduzir.extend(outros_itens_a_deduzir)

        # Perguntar sobre itens a acrescentar
        itens_a_acrescentar = []
        if questionary.confirm("\nHá pesos a serem ACRESCENTADOS para a condição final (leve)?", default=False).ask():
            itens_a_acrescentar = self._obter_lista_de_itens("Itens a Acrescentar")

        # Obter os dados brutos dos pêndulos ou tubos em U
        dados_leituras = self._obter_dados_leituras_inclinacao(metodo_inclinacao)

        # Junta todas as informações num único dicionário e retorna
        dados_finais_rpi = {
            "metodo_inclinacao": metodo_inclinacao,
            "tipo_pesos": tipo_pesos,
            "dados_flutuacao": dados_flutuacao,
            "densidades_medidas": densidades_medidas,
            "dados_tanques": lista_tanques,
            "itens_a_deduzir": itens_a_deduzir,
            "itens_a_acrescentar": itens_a_acrescentar,
            "dados_leituras": dados_leituras
        }

        return dados_finais_rpi
    
    def _obter_dados_leituras_inclinacao(self, metodo_inclinacao: str) -> Dict[str, Any]:
        """
        Método auxiliar para recolher os dados brutos dos pêndulos ou tubos em U.
        """
        dados_leituras = {}
        # --- PÊNDULOS ---
        if "Pêndulos" in metodo_inclinacao:
            num_dispositivos = int(questionary.text(
                "Quantos pêndulos foram utilizados?",
                default="3",
                validate=self._validar_int_min_dois
            ).ask())
            
            lista_pendulos = []
            for i in range(num_dispositivos):
                print(f"\n--- Dados para o Pêndulo nº {i+1}/{num_dispositivos} ---")
                comprimento = float(questionary.text("Comprimento do pêndulo [m]:", validate=self._validar_float_positivo).ask())
                leituras_movimentos = []
                for mov in range(9): # Movimento 0 (inicial) + 8 movimentos
                    print(f"  --- Leituras para o Movimento nº {mov} ---")
                    leituras = questionary.form(
                        maximos=questionary.text("5 leituras MÁXIMAS (separadas por ';'):", validate=self._validar_lista_com_5_numeros),
                        minimos=questionary.text("5 leituras MÍNIMAS (separadas por ';'):", validate=self._validar_lista_com_5_numeros),
                    ).ask()
                    if not leituras: print("Operação cancelada."); exit()
                    leituras_movimentos.append(leituras)
                lista_pendulos.append({"comprimento": comprimento, "leituras": leituras_movimentos})
            dados_leituras["pendulos"] = lista_pendulos

        # --- TUBOS EM U ---
        else:
            num_dispositivos = int(questionary.text(
                "Quantos tubos em U foram utilizados?",
                default="3",
                validate=self._validar_int_min_dois
            ).ask())

            lista_tubos = []
            for i in range(num_dispositivos):
                print(f"\n--- Dados para o Tubo em U nº {i+1}/{num_dispositivos} ---")
                distancia = float(questionary.text("Distância entre as partes verticais do tubo [m]:", validate=self._validar_float_positivo).ask())
                leituras_movimentos = []
                for mov in range(9): # Movimento 0 (inicial) + 8 movimentos
                    print(f"  --- Leituras para o Movimento nº {mov} ---")
                    leituras = questionary.form(
                        maximos_bb=questionary.text("5 leituras MÁXIMAS em Bombordo (separadas por ';'):", validate=self._validar_lista_com_5_numeros, default="1;1;1;1;1"),
                        minimos_bb=questionary.text("5 leituras MÍNIMAS em Bombordo (separadas por ';'):", validate=self._validar_lista_com_5_numeros, default="1;1;1;1;1"),
                        maximos_be=questionary.text("5 leituras MÁXIMAS em Boreste (separadas por ';'):", validate=self._validar_lista_com_5_numeros, default="1;1;1;1;1"),
                        minimos_be=questionary.text("5 leituras MÍNIMAS em Boreste (separadas por ';'):", validate=self._validar_lista_com_5_numeros, default="1;1;1;1;1"),
                    ).ask()
                    if not leituras: print("Operação cancelada."); exit()
                    leituras_movimentos.append(leituras)
                lista_tubos.append({"distancia_vertical": distancia, "leituras": leituras_movimentos})
            dados_leituras["tubos"] = lista_tubos

        return dados_leituras
    
    def _obter_lista_de_itens(self, titulo_secao: str) -> List[Dict[str, Any]]:
        """
        Método auxiliar genérico para recolher uma lista de itens com peso e C.G.

        Args:
            titulo_secao (str): O título a ser exibido ao utilizador (ex: "Itens a Deduzir").

        Returns:
            List[Dict[str, Any]]: Uma lista de dicionários, onde cada dicionário
                                  representa um item.
        """
        lista_itens = []
        num_itens_str = questionary.text(
            f"Quantos {titulo_secao} há?",
            validate=self._validar_int_positivo # Validador para garantir que é um número > 0
        ).ask()
        num_itens = int(num_itens_str) if num_itens_str else 0

        for i in range(num_itens):
            print(f"\n--- Dados para o {titulo_secao} nº {i+1}/{num_itens} ---")
            dados_item = questionary.form(
                nome=questionary.text("Nome do Item:"),
                peso=questionary.text("Peso [t]:", validate=self._validar_float_positivo),
                lcg=questionary.text("Posição Longitudinal do CG (LCG) [m]:", validate=self._validar_float_qualquer),
                vcg=questionary.text("Posição Vertical do CG (VCG) [m]:", validate=self._validar_float_positivo),
            ).ask()
            if not dados_item: print("Operação cancelada."); exit()
            lista_itens.append(dados_item)
        
        return lista_itens
    
    def _obter_lista_de_tanques(self, titulo_secao: str) -> List[Dict[str, Any]]:
        """
        Método auxiliar para recolher uma lista de tanques com suas dimensões e conteúdo.
        """
        lista_tanques = []
        print(f"\n--- {titulo_secao} ---")
        
        if not questionary.confirm("Deseja adicionar tanques para o cálculo de superfície livre?", default=True).ask():
            return lista_tanques

        num_itens_str = questionary.text(
            "Quantos tanques deseja adicionar?",
            validate=self._validar_int_positivo
        ).ask()
        num_itens = int(num_itens_str) if num_itens_str else 0

        for i in range(num_itens):
            print(f"\n--- Dados para o Tanque nº {i+1}/{num_itens} ---")
            dados_tanque = questionary.form(
                nome=questionary.text("Descrição do Tanque:"),
                volume=questionary.text("Volume de líquido no tanque [m³]:", validate=self._validar_float_positivo),
                comprimento=questionary.text("Comprimento do tanque (L) [m]:", validate=self._validar_float_positivo),
                largura=questionary.text("Largura do tanque (B) [m]:", validate=self._validar_float_positivo),
                altura=questionary.text("Altura do tanque (H) [m]:", validate=self._validar_float_positivo),
                densidade=questionary.text("Densidade do líquido [t/m³]:", validate=self._validar_float_positivo, default="1.025")
            ).ask()
            if not dados_tanque: 
                print("Operação cancelada.")
                exit()
            lista_tanques.append(dados_tanque)
        
        return lista_tanques

    def obter_dados_estudo_estabilidade(self, navio_leve_resultados: Dict[str, Any]) -> Dict[str, Any]:
        """
        Coleta os dados para o estudo de estabilidade, começando com os dados da
        embarcação leve e depois pedindo os itens de carga.

        Args:
            navio_leve_resultados (Dict[str, Any]): Dicionário com os resultados
                                                    do navio leve (peso, lcg, vcg)
                                                    calculados no módulo RPI.
        """
        print("\n--- Entrada de Dados para o Estudo de Estabilidade ---")

        # 1. INICIALIZA A TABELA DE PESOS COM OS DADOS DA EMBARCAÇÃO LEVE
        # A tabela agora começa com o item mais importante.
        tabela_pesos_centros = [{
            'categoria': 'EMBARCAÇÃO LEVE',
            'nome': 'Peso Leve',
            'peso': str(navio_leve_resultados.get('Deslocamento Leve (t)', 0.0)),
            'lcg': str(navio_leve_resultados.get('LCG Leve (m)', 0.0)),
            'vcg': str(navio_leve_resultados.get('VCG Leve (m)', 0.0))
        }]
        
        print("\nDados da Embarcação Leve (calculados no RPI) adicionados como base:")
        # Usamos pandas para exibir de forma bonita
        import pandas as pd
        print(pd.DataFrame([tabela_pesos_centros[0]]).to_string(index=False))
        
        tabela_pesos_centros = []
        CATEGORIAS = [
            "CARGAS",
            "PASSAGEIROS E BAGAGENS",
            "TRIPULAÇÃO",
            "GENEROS E ÓLEOS"
        ]

        confirmou_passageiros = False

        # Loop através de cada categoria definida
        for categoria in CATEGORIAS:
            print(f"\n--- Categoria: {categoria} ---")
            if questionary.confirm(f"Deseja adicionar itens nesta categoria?", default=True).ask():
                num_itens_str = questionary.text(
                    f"Quantos itens para a categoria '{categoria}'?",
                    validate=self._validar_int_positivo
                ).ask()

                if categoria == "PASSAGEIROS E BAGAGENS":
                    confirmou_passageiros = True

                num_itens = int(num_itens_str) if num_itens_str else 0

                for i in range(num_itens):
                    print(f"\n--- Item nº {i+1}/{num_itens} da categoria '{categoria}' ---")
                    dados_item = questionary.form(
                        nome=questionary.text("Descrição do Item:"),
                        peso=questionary.text("Peso [t]:", validate=self._validar_float_positivo),
                        lcg=questionary.text("Posição Longitudinal (LCG) [m]:", validate=self._validar_float_qualquer),
                        vcg=questionary.text("Posição Vertical (VCG) [m]:", validate=self._validar_float_positivo),
                    ).ask()
                    
                    if not dados_item: 
                        print("Operação cancelada.")
                        exit()
                    
                    # Adiciona a categoria ao dicionário do item antes de salvá-lo
                    dados_item['categoria'] = categoria
                    tabela_pesos_centros.append(dados_item)
        
        ponto_alagamento = {}
        # Pergunta se existe um ponto de alagamento
        if questionary.confirm("\nHá algum ponto de alagamento a ser considerado (ex: respiro de tanque, porta)?", default=False).ask():
            print("\n--- Dados do Ponto de Alagamento ---")
            dados_ponto = questionary.form(
                descricao=questionary.text("Descrição do ponto de alagamento:"),
                x=questionary.text("Posição Longitudinal (X) do ponto [m]:", validate=self._validar_float_qualquer),
                y=questionary.text("Posição Transversal (Y) do ponto [m]:", validate=self._validar_float_positivo),
                z=questionary.text("Posição Vertical (Z) do ponto [m]:", validate=self._validar_float_positivo)
            ).ask()
            
            if not dados_ponto: 
                print("Operação cancelada.")
                exit()
            ponto_alagamento = dados_ponto

        else:
            # Valores padrão caso não haja ponto de alagamento
            ponto_alagamento = {
                "descricao": "Nenhum",
                "x": 0.0,
                "y": 0.0,
                "z": 0.0
            }
        
        dados_passageiros = []
        
        if confirmou_passageiros:
            print("\n--- Detalhamento da Distribuição de Passageiros por Convés ---")
            num_conveses_str = questionary.text(
                "Em quantos conveses (decks) os passageiros estão distribuídos?",
                validate=self._validar_int_positivo
            ).ask()
            
            if not num_conveses_str: 
                print("Operação cancelada.")
                exit()
            num_conveses = int(num_conveses_str)

            for i in range(num_conveses):
                print(f"\n--- Informações para o Convés de Passageiros nº {i+1}/{num_conveses} ---")
                dados_conves = questionary.form(
                    num_passageiros=questionary.text(
                        "Número de passageiros neste convés:",
                        validate=self._validar_int_positivo
                    ),
                    dist_cl=questionary.text(
                        "Distância do centroide dos passageiros à Linha de Centro (Y) [m]:",
                        validate=self._validar_float_positivo
                    )
                ).ask()
                if not dados_conves: 
                    print("Operação cancelada.")
                    exit()
                
                # Adiciona um identificador do convés para clareza futura
                dados_conves['conves_id'] = f"Convés {i+1}"
                dados_passageiros.append(dados_conves)
        
        # Se o usuário responder "Não", a lista dados_passageiros permanecerá vazia.

        dados_tanques = self._obter_lista_de_tanques(
            titulo_secao="Tanques para Análise de Superfície Livre"
        )

        print("\n--- Dados para os Cálculos de Momentos Emborcadores ---")
        dados_momentos = questionary.form(
            velocidade_operacao=questionary.text(
                "Velocidade de operação (Vo) [nós]:",
                validate=self._validar_float_positivo,
                default="10"
            )
        ).ask()
        if not dados_momentos:
            print("Operação cancelada.")
            exit()
        
        # O retorno da função agora inclui todos os dados coletados
        return {
            "tabela_pesos": tabela_pesos_centros,
            "ponto_alagamento": ponto_alagamento,
            "dados_passageiros": dados_passageiros,
            "dados_tanques": dados_tanques,
            **dados_momentos
        }
    
    def obter_dados_vento_condicao(self, nome_condicao: str, hidro_condicao: Dict[str, Any]) -> Dict[str, Any]:
        """
        Pergunta ao usuário os dados de vento para uma condição de carga específica,
        mostrando os calados como referência.
        """
        print(f"\n--- Dados de Vento para a Condição: '{nome_condicao}' ---")
        print("Use os calados de referência abaixo para determinar os valores nos seus desenhos:")
        print(f"  - Calado na PP de Ré:   {hidro_condicao.get('Calado Ré (m)', 0.0):.3f} m")
        print(f"  - Calado na PP de Vante: {hidro_condicao.get('Calado Vante (m)', 0.0):.3f} m")
        
        dados_vento = questionary.form(
            area_velica=questionary.text(
                "Área vélica lateral (A) nesta condição [m²]:",
                validate=self._validar_float_positivo,
                default="50"
            ),
            distancia_h=questionary.text(
                "Distância vertical (h) do centro da área vélica à linha d'água [m]:",
                validate=self._validar_float_positivo,
                default="3.0"
            )
        ).ask()
        if not dados_vento:
            print("Operação cancelada.")
            exit()
        
        return dados_vento

    def obter_area_navegacao(self) -> str:
        """
        Pergunta ao usuário a área de navegação da embarcação.
        """
        print("\n" + "="*60)
        print("--- VERIFICAÇÃO FINAL DOS CRITÉRIOS DE ESTABILIDADE ---")
        area = questionary.select(
            "Qual a área de navegação da embarcação para esta análise?",
            choices=["Área 1", "Área 2", "Área 1 e Área 2 (aplicará o mais rigoroso)"],
            default="Área 2"
        ).ask()
        if not area:
            print("Operação cancelada.")
            exit()
        return area

    def obter_caminho_salvar(
        self, tipo_resultado: str, nome_arquivo_padrao: str, nome_projeto: str
    ) -> Optional[str]:
        """
        Pergunta ao utilizador se deseja salvar um resultado e obtém o caminho.

        Args:
            tipo_resultado (str): O nome do resultado a ser salvo (ex: "hidrostáticos").
            nome_arquivo_padrao (str): O nome do ficheiro sugerido como padrão.

        Returns:
            Optional[str]: O caminho do ficheiro confirmado pelo utilizador, ou None se
                           o utilizador optar por não salvar.
        """
        caminho_salvar = None
        if questionary.confirm(f"Deseja salvar os resultados {tipo_resultado} em um ficheiro CSV?").ask():
            default_path = os.path.join("data", "projetos_salvos", nome_projeto, nome_arquivo_padrao)
            path_sugerido = questionary.path(
                f"Digite o caminho para salvar o CSV ({tipo_resultado}):",
                default=default_path,
                validate=lambda p: p.lower().endswith('.csv'),
                file_filter=lambda p: p.lower().endswith('.csv')
            ).ask()

            if path_sugerido:
                if questionary.confirm(f"Salvar os resultados em '{path_sugerido}'?").ask():
                    caminho_salvar = path_sugerido
            
            if not caminho_salvar:
                 print("Opção de salvar cancelada.")

        return caminho_salvar