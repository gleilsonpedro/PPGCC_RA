from data.load_datasets import carregar_dataset
from models.train_model import treinar_modelo
from explanations.pi_explanation import analisar_instancias, contar_features_relevantes
from explanations.shap_explanation import calcular_shap
from explanations.lime_explanation import calcular_lime
import time
import os

# Função para limpar o terminal
def limpar_terminal():
    if os.name == 'nt':  # Windows
        os.system('cls')
    else:  # Linux/Mac
        os.system('clear')

# Menu de seleção de datasets
menu = '''
|  ************************* MENU ***************************  |
|  0 - iris                     |  1 - wine                     |
|  2 - breast_cancer            |  3 - digits                  |
|  4 - banknote_authentication  |  5 - wine_quality           |
|  6 - heart_disease            |  7 - parkinsons             |
|  8 - car_evaluation           |  9 - diabetes_binary        |
|  Q - SAIR                                                |
|-------------------------------------------------------------|
'''

# Exibe o menu e solicita uma escolha
print(menu)
opcao = input("Digite o número do dataset ou 'Q' para sair: ").upper().strip()

# Processa a opção selecionada

if opcao == 'Q':
    print("Você escolheu sair.")
  
elif opcao.isdigit() and 0 <= int(opcao) <= 9:
    nomes_datasets = [
        'iris', 'wine', 'breast_cancer', 'digits', 'banknote_authentication',
        'wine_quality', 'heart_disease', 'parkinsons', 'car_evaluation', 'diabetes_binary'
    ]
    nome_dataset = nomes_datasets[int(opcao)]
    
    # Limpa o terminal após a escolha do dataset
    limpar_terminal()
    
    print(f"Dataset '{nome_dataset}' escolhido.")
    try:
        # Carrega o dataset
        X, y, class_names = carregar_dataset(nome_dataset)
        print(f"Dataset {nome_dataset} carregado com sucesso.")
        print("Classes:", class_names)
        print("Amostras:", X.shape[0], "| Atributos:", X.shape[1])
        
        # Medindo o tempo de treinamento do modelo
        inicio_treinamento = time.time()
        modelo, X_test, y_test = treinar_modelo(X, y, classe_0=0)
        fim_treinamento = time.time()
        tempo_treinamento = fim_treinamento - inicio_treinamento
        
        # Medindo o tempo de cálculo das PI-explicações
        inicio_pi = time.time()
        TUDO = analisar_instancias(X_test, y_test, class_names, modelo, X)
        fim_pi = time.time()
        tempo_pi = fim_pi - inicio_pi
        
        # Calcula SHAP
        print("\nCalculando explicações SHAP...")
        inicio_shap = time.time()
        shap_values = calcular_shap(modelo, X_test)
        fim_shap = time.time()
        tempo_shap = fim_shap - inicio_shap
        
        # Calcula LIME
        print("\nCalculando explicações LIME...")
        inicio_lime = time.time()
        lime_values = calcular_lime(modelo, X, X_test, y)
        fim_lime = time.time()
        tempo_lime = fim_lime - inicio_lime
        
        # Tempo total
        tempo_total = tempo_treinamento + tempo_pi + tempo_shap + tempo_lime
        
        # Exibe os tempos de execução
        print()
        print(f"Tempo de treinamento do modelo: {tempo_treinamento:.4f} segundos")
        print(f"Tempo de cálculo das PI-explicações: {tempo_pi:.4f} segundos")
        print(f"Tempo de cálculo das SHAP-explicações: {tempo_shap:.4f} segundos")
        print(f"Tempo de cálculo das LIME-explicações: {tempo_lime:.4f} segundos")
        print(f"Tempo total de execução: {tempo_total:.4f} segundos\n")
        # Conta as features relevantes
        print("Contagem de features relevantes:")
        contar_features_relevantes(TUDO)
        
      
    except Exception as e:
        print(f"Erro ao processar o dataset: {e}")
else:
    # Limpa o terminal antes de exibir o menu novamente
    limpar_terminal()
    print(menu)
    print("Opção inválida. Por favor, escolha um número do menu ou 'Q' para sair.")
    opcao = input("Digite o número do dataset ou 'Q' para sair: ").upper().strip()
