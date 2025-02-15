# main.py

from data.load_datasets import carregar_dataset
from models.train_model import treinar_modelo
from explanations.pi_explanation import analisar_instancias, contar_features_relevantes

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
while True:
    if opcao == 'Q':
        print("Você escolheu sair.")
        break
    elif opcao.isdigit() and 0 <= int(opcao) <= 9:
        nomes_datasets = [
            'iris', 'wine', 'breast_cancer', 'digits', 'banknote_authentication',
            'wine_quality', 'heart_disease', 'parkinsons', 'car_evaluation', 'diabetes_binary'
        ]
        nome_dataset = nomes_datasets[int(opcao)]
        print(f"Dataset '{nome_dataset}' escolhido.")
        try:
            X, y, class_names = carregar_dataset(nome_dataset)
            print(f"Dataset {nome_dataset} carregado com sucesso.")
            print("Classes:", class_names)
            print("Amostras:", X.shape[0], "| Atributos:", X.shape[1])

            # Treina o modelo
            modelo, X_test, y_test = treinar_modelo(X, y, classe_0=0)

            # Analisa as instâncias e captura a lista TUDO
            TUDO = analisar_instancias(X_test, y_test, class_names, modelo, X)

            # Conta as features relevantes
            contar_features_relevantes(TUDO)
            break
        except Exception as e:
            print(f"Erro ao processar o dataset: {e}")
    else:
        print(menu)
        print("Opção inválida. Por favor, escolha um número do menu ou 'Q' para sair.")
        opcao = input("Digite o número do dataset ou 'Q' para sair: ").upper().strip()