import questionary
import os
from typing import Dict, Any

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

    def _validar_lista_calados(self, text: str) -> bool:
        """Validador para a string de calados separados por ';'."""
        if not text:
            return False
        try:
            [float(c.strip()) for c in text.split(';')]
            return True
        except ValueError:
            return False

    def obter_dados_entrada(self) -> Dict[str, Any]:
        """
        Coleta todos os dados de entrada necessários do usuário de forma interativa.
        """
        print("--- TCC Naval - Entrada de Dados ---")
        print("Por favor, forneça as informações a seguir:\n")

        # 1. Obter o caminho do arquivo CSV
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
            calados_str = questionary.text("Digite os calados separados por ponto e vírgula (ex: 0.5; 1.0; 1.5):", validate=self._validar_lista_calados).ask()
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

        # 5. Perguntar se deseja salvar os resultados
        caminho_salvar = None
        if questionary.confirm("Deseja salvar a tabela de resultados em um arquivo CSV?").ask():
            path_salvar = ""
            while not path_salvar:
                # Sugere um caminho e nome de arquivo padrão
                default_path = os.path.join(os.getcwd(), 'data', 'projetos_salvos', 'resultados_hidrostaticos.csv')
                
                path_sugerido = questionary.path(
                    "Digite o caminho e o nome do arquivo para salvar (ex: resultados.csv):",
                    default=default_path,
                    validate=lambda p: p.lower().endswith('.csv'),
                    file_filter=lambda p: p.lower().endswith('.csv')
                ).ask()

                if path_sugerido:
                    confirmado = questionary.confirm(f"Salvar os resultados em '{path_sugerido}'?").ask()
                    if confirmado:
                        path_salvar = path_sugerido
                else:
                    # Usuário cancelou a digitação do caminho
                    print("Opção de salvar cancelada.")
                    break # Sai do while
            
            caminho_salvar = path_salvar

        # 5. Unir todos os dados em um único dicionário
        dados_finais = {
            "caminho_arquivo": caminho_arquivo,
            **dados_embarcacao,
            "metodo_interp": metodo_interp,
            "referencial": referencial,
            "calados": dados_calado,
            "caminho_salvar": caminho_salvar, 
        }
        
        return dados_finais