import questionary
import os
from typing import Dict, Any, Optional
import pandas as pd

class Menu:
    """
    Gerencia a interface do usuário no terminal para a entrada de dados.
    """

    def _validar_float(self, text: str) -> bool:
        """Validador para garantir que a entrada seja um float positivo ou zero."""
        if not text:
            return False
        try:
            value = float(text)
            return value >= 0
        except ValueError:
            return False

    def _validar_int(self, text: str) -> bool:
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
        
    def obter_escolha_calculo(self) -> str:
        """Pergunta ao utilizador qual o principal tipo de cálculo a ser realizado."""
        escolha = questionary.select(
            "Qual cálculo deseja realizar?",
            choices=[
                "1. Apenas Curvas Hidrostáticas",
                "2. Curvas Cruzadas de Estabilidade (KN)",
            ]
        ).ask()
        return escolha

    def obter_dados_hidrostaticos(self) -> Dict[str, Any]:
        """
        Coleta todos os dados de entrada necessários do usuário de forma interativa.
        """
        print("--- TCC Naval - Entrada de Dados ---")
        print("Por favor, forneça as informações a seguir:\n")

        # 1. Obtem o nome da embarcação/projeto e o caminho do arquivo CSV
        nome_projeto = questionary.text("Qual o nome do projeto ou embarcação?").ask()
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
            lpp=questionary.text("Comprimento entre perpendiculares (Lpp) em metros:", validate=self._validar_float, default="19.713"),
            boca=questionary.text("Boca moldada em metros:", validate=self._validar_float, default="6"),
            pontal=questionary.text("Pontal moldado em metros:", validate=self._validar_float, default="3"),
            densidade=questionary.text("Densidade da água (ex: 1.025):", default="1.025", validate=self._validar_float)
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
                calado_min=questionary.text("Calado mínimo:", validate=lambda val: self._validar_float(val) and float(val) < float(dados_embarcacao['pontal'])),
                calado_max=questionary.text("Calado máximo:", validate=lambda val: self._validar_float(val) and float(val) <= float(dados_embarcacao['pontal']))
            ).ask()

            if not calados_min_max:
                print("Operação cancelada."); exit()
            
            calado_min = float(calados_min_max['calado_min'])
            calado_max = float(calados_min_max['calado_max'])

            if calado_min >= calado_max:
                print("\nErro: O calado mínimo deve ser menor que o calado máximo.")
                exit()

            if "número" in metodo_calado:
                num_calados = int(questionary.text("Número de calados:", validate=self._validar_int).ask())
                dados_calado = {"metodo": "numero", "min": calado_min, "max": calado_max, "num": num_calados}
            
            else: # Opção de passo
                passo = float(questionary.text("Passo entre os calados:", validate=lambda val: self._validar_float(val) and float(val) <= (calado_max - calado_min)).ask())
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
        print(f"INFO: O deslocamento máximo da hidrostática é {max_desloc:.2f} t.")

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
                deslocamento_min=questionary.text("Deslocamento mínimo:", validate=lambda val: self._validar_float(val)),
                deslocamento_max=questionary.text("Deslocamento máximo:", validate=lambda val: self._validar_float(val))
            ).ask()

            if not deslocamentos_min_max:
                print("Operação cancelada."); exit()
            
            deslocamento_min = float(deslocamentos_min_max['deslocamento_min'])
            deslocamento_max = float(deslocamentos_min_max['deslocamento_max'])

            if deslocamento_min >= deslocamento_max:
                print("\nErro: O deslocamento mínimo deve ser menor que o deslocamento máximo.")
                exit()

            if "número" in metodo_deslocamento:
                num_deslocamentos = int(questionary.text("Número de deslocamentos:", validate=self._validar_int).ask())
                dados_deslocamento = {"metodo": "numero", "min": deslocamento_min, "max": deslocamento_max, "num": num_deslocamentos}
            
            else: # Opção de passo
                passo = float(questionary.text("Passo entre os deslocamentos:", validate=lambda val: self._validar_float(val) and float(val) <= (deslocamento_max - deslocamento_min)).ask())
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
                angulo_min=questionary.text("Ângulo mínimo:", validate=lambda val: self._validar_float(val)),
                angulo_max=questionary.text("Ângulo máximo:", validate=lambda val: self._validar_float(val))
            ).ask()

            if not angulos_min_max:
                print("Operação cancelada."); exit()
            
            angulo_min = float(angulos_min_max['angulo_min'])
            angulo_max = float(angulos_min_max['angulo_max'])

            if angulo_min >= angulo_max:
                print("\nErro: O ângulo mínimo deve ser menor que o angulo máximo.")
                exit()

            if "número" in metodo_angulo:
                num_angulos = int(questionary.text("Número de ângulos:", validate=self._validar_int).ask())
                dados_angulo = {"metodo": "numero", "min": angulo_min, "max": angulo_max, "num": num_angulos}
            
            else: # Opção de passo
                passo = float(questionary.text("Passo entre os ângulos:", validate=lambda val: self._validar_float(val) and float(val) <= (angulo_max - angulo_min)).ask())
                dados_angulo = {"metodo": "passo", "min": angulo_min, "max": angulo_max, "passo": passo}
            
        dados_finais = {"deslocamentos": dados_deslocamento, "angulos": dados_angulo}
            
        return dados_finais
    
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