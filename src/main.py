from .ui.menu import Menu
from .io.file_handler import FileHandler
from .utils.calado_utils import gerar_lista_de_calados
from .core.chc import InterpoladorCasco, CalculadoraHidrostatica
import pandas as pd

def main():
    """
    Função principal que orquestra a execução do programa.
    """
    print("Iniciando aplicação naval TCC...")
    
    # 1. Coletar dados do usuário
    menu_principal = Menu()
    try:
        dados_de_entrada = menu_principal.obter_dados_entrada()
        lpp = float(dados_de_entrada['lpp'])
        densidade = float(dados_de_entrada['densidade'])
        
    except (KeyboardInterrupt, TypeError):
        print("\nPrograma encerrado pelo usuário.")
        return

    # 2. Ler e Processar o arquivo da tabela de cotas
    manipulador_arquivos = FileHandler()
    try:
        # Lê o arquivo CSV bruto
        tabela_bruta = manipulador_arquivos.ler_tabela_cotas(
            dados_de_entrada['caminho_arquivo']
        )
        print(f"\nArquivo '{dados_de_entrada['caminho_arquivo']}' lido com sucesso.")
        
        # Processa e valida os dados
        tabela_processada = manipulador_arquivos.processar_dados_balizas(
            tabela_bruta,
            lpp,
            dados_de_entrada['referencial']
        )
        
        print("\nTabela de Cotas Processada:")
        pd.set_option('display.max_rows', 20) # Limita a exibição para não poluir o terminal
        print(tabela_processada)

    except (FileNotFoundError, ValueError) as e:
        print(f"\nErro no processamento de dados: {e}")
        print("A aplicação será encerrada.")
        return
    
    # 3. Gerar lista de calados e preparar para o cálculo
    lista_calados = gerar_lista_de_calados(dados_de_entrada['calados'])
    if not lista_calados:
        print("Nenhum calado válido foi gerado. Encerrando.")
        return

    print(f"Calados a serem calculados: {[round(c, 3) for c in lista_calados]}")

    # 4. Executar os cálculos hidrostáticos


    # --- Próximos Passos ---
    # 5. Processar os dados dos calados para gerar a lista final de calados
    # 6. Executar os cálculos hidrostáticos (usando o módulo core/chc.py)
    # 7. Exibir ou salvar os resultados

    print("\nAplicação finalizada por enquanto.")


if __name__ == '__main__':
    main()