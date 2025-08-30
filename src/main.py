from .ui.menu import Menu
from .io.file_handler import FileHandler
from .core.ch import InterpoladorCasco, CalculadoraHidrostatica
from .core.cc import CalculadoraCurvasCruzadas
from .utils.list_utils import gerar_lista_de_calados, gerar_lista_deslocamentos, gerar_lista_angulos
from .ui.display import exibir_tabela_hidrostatica

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


def main():
    """
    Função principal que orquestra a execução do programa.
    """
    print("Iniciando aplicação naval TCC...")

    menu = Menu()
    manipulador_arquivos = FileHandler() # Instanciar uma vez no início

    # 1. Perguntar qual cálculo fazer
    escolha = menu.obter_escolha_calculo()

    # 2. Obter dados básicos e calcular a hidrostática
    dados_hidrostaticos = menu.obter_dados_hidrostaticos()

    # 3. Obtem os dados de entrada
    try:
        nome_projeto = dados_hidrostaticos['nome_projeto']
        lpp = float(dados_hidrostaticos['lpp'])
        densidade = float(dados_hidrostaticos['densidade'])
        metodo_interp = dados_hidrostaticos['metodo_interp']
        
    except (KeyboardInterrupt, TypeError):
        print("\nPrograma encerrado pelo usuário.")
        return

    # 4. Ler e Processar a tabela de cotas
    try:
        tabela_bruta = manipulador_arquivos.ler_tabela_cotas(
            dados_hidrostaticos['caminho_arquivo']
        )
        tabela_processada = manipulador_arquivos.processar_dados_balizas(
            tabela_bruta,
            lpp,
            dados_hidrostaticos['referencial']
        )
        print("\n-> Tabela de Cotas lida e validada com sucesso.")

    except (FileNotFoundError, ValueError) as e:
        print(f"\nErro no processamento de dados: {e}")
        return

    # 5. Gerar lista de calados
    lista_calados = gerar_lista_de_calados(dados_hidrostaticos['calados'])
    if not lista_calados:
        print("Nenhum calado válido foi gerado. Encerrando.")
        return
    
    # 6. Executar os cálculos hidrostáticos
    try:
        casco_interpolado = InterpoladorCasco(
            tabela_processada, metodo_interp=metodo_interp
        )
        
        calculadora = CalculadoraHidrostatica(
            casco_interpolado, densidade=densidade
        )
        
        # Executa o cálculo para todos os calados e obtém o DataFrame final
        df_hidrostatico = calculadora.calcular_curvas(lista_calados)
        
        # 7. Exibir os resultados na tabela estilizada
        exibir_tabela_hidrostatica(df_hidrostatico)

    except Exception as e:
        print(f"\nOcorreu um erro inesperado durante os cálculos: {e}")
        import traceback
        traceback.print_exc()
    
    # 8. Perguntar se quer salvar os resultados hidrostáticos
    caminho_salvar_hidro = menu.obter_caminho_salvar(
        tipo_resultado="hidrostáticos",
        nome_arquivo_padrao="resultados_hidrostaticos.csv",
        nome_projeto=nome_projeto
    )
    if caminho_salvar_hidro:
        manipulador_arquivos.salvar_resultados_csv(df_hidrostatico, caminho_salvar_hidro)

    # 9. Se a escolha foi apenas hidrostática, termina aqui.
    if "Apenas" in escolha:
        print("\nCálculo de Curvas Hidrostáticas concluído.")
        return

    # 10. Se a escolha foi Curvas Cruzadas, continuar
    dados_kn = menu.obter_dados_curvas_cruzadas(df_hidrostatico)
    
    # 11. Gerar listas de deslocamentos e ângulos
    lista_deslocamentos = gerar_lista_deslocamentos(dados_kn["deslocamentos"])
    lista_angulos = gerar_lista_angulos(dados_kn["angulos"])

    print(f"\n-> Gerados {len(lista_deslocamentos)} deslocamentos e {len(lista_angulos)} ângulos para os cálculos de KN.")
    
    # 12. Criar e executar a calculadora de curvas cruzadas
    calculadora_kn = CalculadoraCurvasCruzadas(casco_interpolado, df_hidrostatico, dados_hidrostaticos)
    df_kn = calculadora_kn.calcular_curvas_kn(lista_deslocamentos, lista_angulos)

    # 13. Exibir e salvar os resultados de KN
    exibir_tabela_hidrostatica(df_kn) # Reutilizamos o mesmo formatador de tabela
    
    # 14. Perguntar se quer salvar os resultados de KN
    caminho_salvar_kn = menu.obter_caminho_salvar(
        tipo_resultado="das curvas KN",
        nome_arquivo_padrao="resultados_kn.csv",
        nome_projeto=nome_projeto
    )
    if caminho_salvar_kn:
        # Salvar com o índice para manter o formato de deslocamento nas linhas
        manipulador_arquivos.salvar_resultados_csv(df_kn, caminho_salvar_kn)

    print("\nCálculo de Curvas Cruzadas concluído.")

if __name__ == '__main__':
    main()