from .ui.menu import Menu
from .io.file_handler import FileHandler
from .core.chc import InterpoladorCasco, PropriedadesHidrostaticas
from .utils.calado_utils import gerar_lista_de_calados
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
    
    # --- BLOCO DE TESTE ---
    # Vamos testar o cálculo apenas para o primeiro calado da lista.
    primeiro_calado = 1.98
    print(f"\n--- INICIANDO TESTE PARA O CALADO T = {primeiro_calado:.3f} m ---")
    
    try:
        # 4. Criar o interpolador do casco
        print("-> Criando objeto InterpoladorCasco...")
        casco_interpolado = InterpoladorCasco(tabela_processada, metodo_interp=metodo_interp)
        print("-> Objeto InterpoladorCasco criado.")

        # 5. Criar a instância de PropriedadesHidrostaticas
        print(f"-> Criando objeto PropriedadesHidrostaticas para T={primeiro_calado:.3f}m...")
        props_calado = PropriedadesHidrostaticas(casco_interpolado, primeiro_calado, densidade)
        print("-> Objeto PropriedadesHidrostaticas criado.")
        
        # 6. Executar o cálculo das dimensões da linha d'água
        print("-> Executando _calcular_dimensoes_linha_dagua()...")
        props_calado._calcular_dimensoes_linha_dagua()
        print("-> Cálculo de dimensões finalizado.")

        # 7. Exibir os resultados do teste
        print("\n--- RESULTADOS DO TESTE ---")
        print(f"  Calado (T): {props_calado.calado:.3f} m")
        print(f"  Extremidade de Ré (x_re): {props_calado.x_re:.3f} m")
        print(f"  Extremidade de Vante (x_vante): {props_calado.x_vante:.3f} m")
        print(f"  LWL: {props_calado.lwl:.3f} m")
        print(f"  BWL: {props_calado.bwl:.3f} m")
        print(f"  Área do Plano de Flutuação (AWP): {props_calado.area_plano_flutuacao:.3f} m²")
        print(f"  Volume de Carena: {props_calado.volume:.3f} m³")
        print(f"  Deslocamento: {props_calado.deslocamento:.3f} t")
        print("\nTeste concluído com sucesso!")

    except Exception as e:
        print(f"\nOcorreu um erro inesperado durante o teste: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()