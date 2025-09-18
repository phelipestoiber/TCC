from .ui.menu import Menu
from .io.file_handler import FileHandler
from .core.ch import InterpoladorCasco, CalculadoraHidrostatica
from .core.cc import CalculadoraCurvasCruzadas
from .core.rpi import CalculadoraRPI
from .core.eed import CalculadoraEED, VerificadorCriterios
from .utils.list_utils import gerar_lista_de_calados, gerar_lista_deslocamentos, gerar_lista_angulos
from .ui.display import exibir_tabela_hidrostatica
from .ui.display import exibir_tabela_hidrostatica
from .ui.plotting import plotar_curva_estabilidade

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import pandas as pd

def main():
    """
    Função principal que orquestra a execução do programa.
    """
    print("Iniciando aplicação naval TCC...")

    menu = Menu()
    manipulador_arquivos = FileHandler() # Instanciar uma vez no início

    # 1. Perguntar qual cálculo fazer
    print("--- MÓDULO 1: CURVAS HIDROSTÁTICAS ---")

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
    if not menu.deseja_prosseguir("Cálculo de Curvas Cruzadas"):
        print("\nPrograma finalizado.")
        return

    # 10. Se a escolha foi Curvas Cruzadas, continuar
    print("\n--- MÓDULO 2: CURVAS CRUZADAS DE ESTABILIDADE (KN) ---")
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

    if not menu.deseja_prosseguir("Cálculo do Relatório de Prova de Inclinação (RPI)"):
        print("\nPrograma finalizado.")
        return
    
    print("\n--- MÓDULO 3: RELATÓRIO DE PROVA DE INCLINAÇÃO (RPI) ---")
    dados_rpi = menu.obter_dados_rpi()

    # Inicializa a variável para garantir que ela exista mesmo se o 'try' falhar
    navio_leve_resultados = {}

    try:
        # 1. Instanciar e executar os cálculos do RPI em sequência
        calculadora_rpi = CalculadoraRPI(dados_rpi, dados_hidrostaticos)
        calculadora_rpi.calcular_condicao_flutuacao()
        calculadora_rpi.calcular_densidade_media()
        calculadora_rpi.calcular_pesos_e_momentos()
        calculadora_rpi.processar_leituras_inclinacao()
        calculadora_rpi.calcular_momentos_inclinantes()

        # 2. Exibir os resultados preliminares para verificação (como você já tinha)
        print("\n--- Resultados Preliminares do RPI ---")
        print(f"  Densidade Média: {calculadora_rpi.densidade_media:.4f} t/m³")
        # ... (outros prints de resultados preliminares)

        # 3. PASSO FINAL E CRÍTICO: Calcular a condição de Navio Leve
        # Este método utiliza os resultados dos passos anteriores para o cálculo final
        calculadora_rpi.calcular_condicao_navio_leve()
        navio_leve_resultados = calculadora_rpi.navio_leve_resultados

        # 4. Exibir o resultado final e mais importante do RPI
        print("\n" + "="*50)
        print("--- RESULTADO FINAL DO NAVIO LEVE ---")
        if navio_leve_resultados:
            print(f"  - Deslocamento Leve: {navio_leve_resultados.get('Deslocamento Leve (t)', 0.0):.3f} t")
            print(f"  - LCG Leve: {navio_leve_resultados.get('LCG Leve (m)', 0.0):.3f} m")
            print(f"  - VCG (KG) Leve: {navio_leve_resultados.get('VCG Leve (m)', 0.0):.3f} m")
        else:
            print("  Não foi possível calcular os resultados do navio leve.")
        print("="*50)

    except Exception as e:
        print(f"\nOcorreu um erro durante o cálculo do RPI: {e}")
        # Cria um resultado padrão para poder continuar para o módulo 4 mesmo se o RPI falhar
        navio_leve_resultados = {
            'Deslocamento Leve (t)': 0, 'LCG Leve (m)': 0, 'VCG Leve (m)': 0
        }
        import traceback
        traceback.print_exc()

    # 16. Perguntar se quer salvar os resultados do RPI
    caminho_salvar_rpi = menu.obter_caminho_salvar(
        tipo_resultado="do Relatório Final da Prova de Inclinação",
        nome_arquivo_padrao="relatorio_final_rpi.txt",
        nome_projeto=nome_projeto
    )
    if caminho_salvar_rpi:
        # Substituímos o TODO pela chamada da nossa nova função
        manipulador_arquivos.salvar_relatorio_rpi(caminho_salvar_rpi, calculadora_rpi)

    # 17. Perguntar se deseja prosseguir para o Estudo de Estabilidade
    if not menu.deseja_prosseguir("Cálculo de Estudo de Estabilidade"):
        print("\nPrograma finalizado.")
        return
    
    # 18. Iniciar o Módulo 4: Estudo de Estabilidade
    # --- MÓDULO 4: ESTUDO DE ESTABILIDADE DEFINITIVO (EED) ---
    if not menu.deseja_prosseguir("Cálculo de Estudo de Estabilidade Definitivo"):
        print("\nPrograma finalizado."); return
        
    print("\n--- MÓDULO 4: ESTUDO DE ESTABILIDADE DEFINITIVO ---")
    area_navegacao = menu.obter_area_navegacao()
    dados_estabilidade = menu.obter_dados_estudo_estabilidade(navio_leve_resultados)

    try:
        desloc_leve_final = navio_leve_resultados.get('Deslocamento Leve (t)', 0.0)
        
        calculadora_eed = CalculadoraEED(
            dados_estabilidade=dados_estabilidade, deslocamento_leve=desloc_leve_final,
            casco=casco_interpolado, dados_hidrostaticos=dados_hidrostaticos,
            df_hidrostatico=df_hidrostatico, densidade=densidade
        )
        verificador = VerificadorCriterios(area_navegacao)

        # Executa a cadeia de cálculos do EED
        calculadora_eed.calcular_pesos_e_momentos()
        calculadora_eed.calcular_hidrostaticas_condicoes()
        calculadora_eed.gerar_curvas_estabilidade()

        # Loop final para análise, verificação e plotagem
        print("\n" + "="*80)
        print("--- ANÁLISE FINAL, VERIFICAÇÃO DE CRITÉRIOS E GERAÇÃO DE GRÁFICOS ---")
        
        todos_resultados_criterios = {}
        # caminho_base_graficos = os.path.join("data", "projetos_salvos", nome_projeto, "graficos_eed")

        for nome_cond, dados_cond in calculadora_eed.resultados_condicoes.items():
            if 'hidrostaticos' in dados_cond and 'curva_gz' in dados_cond:
                print(f"\n--- Processando Condição: {nome_cond} ---")
                
                dados_vento = menu.obter_dados_vento_condicao(nome_cond, dados_cond['hidrostaticos'])
                area_velica_cond = float(dados_vento.get('area_velica', 0.0))
                h_vento_cond = float(dados_vento.get('distancia_h', 0.0))
                
                df_gz = dados_cond['curva_gz'].copy()
                
                bracos_emborcadores = []
                for angulo in df_gz['Angulo (°)']:
                    braco_p = calculadora_eed.calcular_momento_passageiros(dados_cond, angulo)
                    braco_g = calculadora_eed.calcular_braco_guinada(dados_cond)
                    braco_v = calculadora_eed.calcular_braco_vento(dados_cond, angulo, area_velica_cond, h_vento_cond)
                    bracos_emborcadores.append(max(braco_g, braco_v) + braco_p)
                df_gz['GZ Emborcador (m)'] = bracos_emborcadores
                
                resultados_criterios = verificador.verificar_todos(dados_cond, df_gz)
                todos_resultados_criterios[nome_cond] = resultados_criterios
                
                print("\n  -> Resultado da Verificação:")
                for criterio, resultado in resultados_criterios.items():
                    status = "PASSOU" if resultado['passou'] else "FALHOU"
                    print(f"    - {criterio:<22} | Valor: {resultado['valor']:<18} | Esperado: {resultado['esperado']:<18} | Status: {status}")

                nome_arquivo_plot = f"curva_estabilidade_{nome_cond.replace(' ', '_').replace(':', '')}.png"
                #caminho_salvar_plot = os.path.join(caminho_base_graficos, nome_arquivo_plot)
                
                plotar_curva_estabilidade(
                    df_curva=df_gz, resultados_criterios=resultados_criterios,
                    dados_condicao=dados_cond, nome_condicao=nome_cond,
                    # caminho_salvar=caminho_salvar_plot
                )
        
        # Salvar o relatório completo
        caminho_salvar_eed = menu.obter_caminho_salvar(
            tipo_resultado="do Relatório Final de Estabilidade (EED)",
            nome_arquivo_padrao="relatorio_final_eed.txt", nome_projeto=nome_projeto
        )
        if caminho_salvar_eed:
            manipulador_arquivos.salvar_relatorio_eed(
                caminho_arquivo=caminho_salvar_eed, calculadora_eed=calculadora_eed,
                verificador=verificador, resultados_verificacao=todos_resultados_criterios
            )

    except Exception as e:
        print(f"\nOcorreu um erro durante o cálculo de estabilidade: {e}")
        import traceback
        traceback.print_exc()

    print("\nAnálise de estabilidade concluída.")
    print("\nPrograma finalizado.")

    

if __name__ == '__main__':
    main()