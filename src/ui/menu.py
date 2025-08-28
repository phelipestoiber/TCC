import questionary
from typing import Dict, Any

class Menu:
    """
    Gerencia a interface do usuário no terminal para a entrada de dados.
    """

    def _validar_float(self, text: str) -> bool:
        """Validador para garantir que a entrada seja um float positivo."""
        if not text:
            return False
        try:
            value = float(text)
            return value >= 0
        except ValueError:
            return False

    def _validar_int(self, text: str) -> bool:
        """Validador para garantir que a entrada seja um inteiro > 2."""
        if not text:
            return False
        try:
            value = int(text)
            return value >= 2
        except ValueError:
            return False

    def _validar_lista_calados(self, text: str) -> bool:
        """Validador para a string de calados separados por ';'."""
        if not text:
            return False
        try:
            # Tenta converter todos os itens para float
            [float(c.strip()) for c in text.split(';')]
            return True
        except ValueError:
            return False

    def obter_dados_entrada(self) -> Dict[str, Any]:
        """
        Coleta todos os dados de entrada necessários do usuário de forma interativa.

        Returns:
            Dict[str, Any]: Um dicionário contendo todos os dados validados.
        """
        print("--- TCC Naval - Entrada de Dados ---")
        print("Por favor, forneça as informações a seguir:\n")

        # 1. Obter o caminho do arquivo CSV
        caminho_arquivo = "C:\\Users\\afmn\\Desktop\\TCC\\data\\exemplos_tabelas_cotas\\TABELA DE COTAS.csv"
        # while not caminho_arquivo:
        #     path = questionary.path(
        #         "Qual o caminho para o arquivo da tabela de cotas (.csv)?",
        #         validate=lambda p: os.path.exists(p) and p.lower().endswith('.csv'),
        #         file_filter=lambda p: p.lower().endswith('.csv')
        #     ).ask()

        #     if path:
        #         confirmado = questionary.confirm(f"O caminho '{path}' está correto?").ask()
        #         if confirmado:
        #             caminho_arquivo = path
        #     else:
        #         # O usuário pressionou Ctrl+C para sair
        #         print("Operação cancelada.")
        #         exit()
        
        # 2. Obter características da embarcação
        dados_embarcacao = questionary.form(
            lpp=questionary.text(
                "Comprimento entre perpendiculares (Lpp) em metros:",
                validate=self._validar_float
            ),
            boca=questionary.text(
                "Boca moldada em metros:",
                validate=self._validar_float
            ),
            pontal=questionary.text(
                "Pontal moldado em metros:",
                validate=self._validar_float
            ),
        ).ask()

        if not dados_embarcacao:
            print("Operação cancelada.")
            exit()

        # 3. Obter configurações de cálculo

        densidade=questionary.text(
            "Densidade da água (ex: 1.025):",
            default="1.025",
            validate=self._validar_float
            )
        
        if not densidade:
            print("Operação cancelada.")
            exit()

        referencial = questionary.select(
            "Qual o referencial para os resultados dos cálculos?",
            choices=[
                "Perpendicular de ré (AP)",
                "Perpendicular de meio-navio (MS)",
            ]
        ).ask()

        if not referencial:
            print("Operação cancelada.")
            exit()

        metodo_interp = questionary.select(
            "Qual método de interpolação deseja usar nos cálculos?",
            choices=[
                "Linear",
                "PCHIP (Cúbica, preserva a monotonia)",
            ],
            instruction="Linear é mais rápido. PCHIP gera curvas mais suaves."
        ).ask()
        
        if not referencial or not metodo_interp:
            print("Operação cancelada.")
            exit()
            
        # 4. Obter a forma de definir os calados
        metodo_calado = questionary.select(
            "Como você deseja definir a faixa de calados para os cálculos?",
            choices=[
                "Fornecer uma lista de calados",
                "Definir calado mínimo, máximo e o número de calados",
                "Definir calado mínimo, máximo e o passo",
            ]
        ).ask()

        if not metodo_calado:
            print("Operação cancelada.")
            exit()

        dados_calado = {}
        if "lista" in metodo_calado:
            calados_str = questionary.text(
                "Digite os calados separados por ponto e vírgula (ex: 0.5; 1.0; 1.5):",
                validate=self._validar_lista_calados
            ).ask()
            dados_calado = {"metodo": "lista", "valores": calados_str}
        
        else: # Opções de min/max
            calado_min = float(questionary.text(
                "Calado mínimo:",
                validate=lambda val: self._validar_float(val) and float(val) < float(dados_embarcacao['pontal'])
            ).ask())

            calado_max = float(questionary.text(
                "Calado máximo:",
                validate=lambda val: self._validar_float(val) and float(val) > calado_min and float(val) <= float(dados_embarcacao['pontal'])
            ).ask())

            if "número" in metodo_calado:
                num_calados = int(questionary.text(
                    "Número de calados:",
                    validate=self._validar_int
                ).ask())
                dados_calado = {"metodo": "numero", "min": calado_min, "max": calado_max, "num": num_calados}
            
            else: # Opção de passo
                passo = float(questionary.text(
                    "Passo entre os calados:",
                    validate=lambda val: self._validar_float(val) and float(val) < (calado_max - calado_min)
                ).ask())
                dados_calado = {"metodo": "passo", "min": calado_min, "max": calado_max, "passo": passo}

        # 4. Unir todos os dados em um único dicionário
        dados_finais = {
            "caminho_arquivo": caminho_arquivo,
            **dados_embarcacao,
            "densidade": densidade,
            "referencial": referencial,
            "calados": dados_calado,
        }
        
        return dados_finais

if __name__ == '__main__':
    menu = Menu()
    try:
        dados_coletados = menu.obter_dados_entrada()
        print("\n--- Dados Coletados com Sucesso ---")
        import json
        print(json.dumps(dados_coletados, indent=4))
    except (KeyboardInterrupt, TypeError):
        print("\nPrograma encerrado pelo usuário.")