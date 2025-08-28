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
    try:
        # Cria o objeto que define a geometria do casco via interpolação
        casco_interpolado = InterpoladorCasco(tabela_processada)
        
        # Cria a calculadora que orquestrará os cálculos
        calculadora = CalculadoraHidrostatica(casco_interpolado, densidade)
        
        # Executa o cálculo para todos os calados
        df_resultados = calculadora.calcular_curvas(lista_calados)
        
        # 5. Exibir os resultados
        print("\n--- CURVAS HIDROSTÁTICAS ---")
        pd.set_option('display.max_rows', 50)
        pd.set_option('display.width', 120) # Ajusta a largura do display
        print(df_resultados.round(4)) # Arredonda para 4 casas decimais

    except Exception as e:
        print(f"\nOcorreu um erro inesperado durante os cálculos: {e}")
        print("Por favor, verifique a consistência da sua tabela de cotas.")

    # --- Próximos Passos ---
    # 3. Processar os dados dos calados para gerar a lista final de calados
    # 4. Executar os cálculos hidrostáticos (usando o módulo core/chc.py)
    # 5. Exibir ou salvar os resultados

    print("\nAplicação finalizada por enquanto.")


if __name__ == '__main__':
    main()