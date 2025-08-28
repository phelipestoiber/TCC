from .ui.menu import Menu
from .io.file_handler import FileHandler
from .core.chc import InterpoladorCasco, CalculadoraHidrostatica
from .utils.calado_utils import gerar_lista_de_calados
from .ui.display import exibir_tabela_hidrostatica
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
        metodo_interp = dados_de_entrada['metodo_interp']
        
    except (KeyboardInterrupt, TypeError):
        print("\nPrograma encerrado pelo usuário.")
        return

    # 2. Ler e Processar a tabela de cotas
    manipulador_arquivos = FileHandler()
    try:
        tabela_bruta = manipulador_arquivos.ler_tabela_cotas(
            dados_de_entrada['caminho_arquivo']
        )
        tabela_processada = manipulador_arquivos.processar_dados_balizas(
            tabela_bruta,
            lpp,
            dados_de_entrada['referencial']
        )
        print("\n-> Tabela de Cotas lida e validada com sucesso.")

    except (FileNotFoundError, ValueError) as e:
        print(f"\nErro no processamento de dados: {e}")
        return

    # 3. Gerar lista de calados
    lista_calados = gerar_lista_de_calados(dados_de_entrada['calados'])
    if not lista_calados:
        print("Nenhum calado válido foi gerado. Encerrando.")
        return
    
    # 4. Executar os cálculos hidrostáticos
    try:
        casco_interpolado = InterpoladorCasco(
            tabela_processada, metodo_interp=dados_de_entrada['metodo_interp']
        )
        
        calculadora = CalculadoraHidrostatica(
            casco_interpolado, densidade=float(dados_de_entrada['densidade'])
        )
        
        # Executa o cálculo para todos os calados e obtém o DataFrame final
        df_resultados = calculadora.calcular_curvas(lista_calados)
        
        # 5. Exibir os resultados na tabela estilizada
        exibir_tabela_hidrostatica(df_resultados)
    
    # 6. Salvar os resultados, se o usuário solicitou
        caminho_salvar = dados_de_entrada.get("caminho_salvar")
        if caminho_salvar:
            manipulador_arquivos.salvar_resultados_csv(df_resultados, caminho_salvar)

    except Exception as e:
        print(f"\nOcorreu um erro inesperado durante os cálculos: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()