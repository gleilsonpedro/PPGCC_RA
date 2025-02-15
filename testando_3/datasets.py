# DATASETS DE CLASSIFICAÇÃO
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_digits
import pandas as pd
from IPython.display import clear_output

# Função para carregar os datasets
def carregar_dataset(nome_dataset):
    if nome_dataset == 'iris':
        data = load_iris()
        X, y = pd.DataFrame(data.data, columns=data.feature_names), data.target
        class_names = data.target_names
    
    elif nome_dataset == 'wine':
        data = load_wine()
        X, y = pd.DataFrame(data.data, columns=data.feature_names), data.target
        class_names = data.target_names
    
    elif nome_dataset == 'breast_cancer':
        data = load_breast_cancer()
        X, y = pd.DataFrame(data.data, columns=data.feature_names), data.target
        class_names = data.target_names
    
    elif nome_dataset == 'digits':
        data = load_digits()
        X, y = pd.DataFrame(data.data, columns=[f"pixel_{i}" for i in range(data.data.shape[1])]), data.target
        class_names = [str(i) for i in range(10)]
    
    elif nome_dataset == 'banknote_authentication':
        data = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt", header=None)
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]
        class_names = ['Legitimate', 'Forgery']
    
    elif nome_dataset == 'wine_quality':
        data = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv", sep=';')
        X = data.drop(columns=['quality'])
        y = data['quality']
        class_names = sorted(y.unique().tolist())
    
    elif nome_dataset == 'heart_disease':
        data = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data", header=None, na_values="?")
        data = data.dropna()  # Remove valores ausentes
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]
        class_names = sorted(y.unique().tolist())
    
    elif nome_dataset == 'parkinsons':
        data = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data")
        X = data.drop(columns=['status', 'name'])
        y = data['status']
        class_names = ['Healthy', 'Parkinsons']
    
    elif nome_dataset == 'car_evaluation':
        data = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data", header=None)
        data.columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
        X = pd.get_dummies(data.drop(columns=['class']))
        y = data['class'].factorize()[0]
        class_names = data['class'].unique()
    
    elif nome_dataset == 'diabetes_binary':
        data = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/diabetes/diabetes_data_upload.csv")
        X = pd.get_dummies(data.drop(columns=['class']))
        y = data['class'].apply(lambda x: 1 if x == 'Positive' else 0)
        class_names = ['Negative', 'Positive']
    
    else:
        raise ValueError("Nome do dataset não reconhecido. Escolha um dataset válido.")
    
    return X, y, class_names

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
        clear_output()
        print("Você escolheu sair.")
        break
    elif opcao.isdigit() and 0 <= int(opcao) <= 9:
        nomes_datasets = [
            'iris', 'wine', 'breast_cancer', 'digits', 'banknote_authentication',
            'wine_quality', 'heart_disease', 'parkinsons', 'car_evaluation', 'diabetes_binary'
        ]
        nome_dataset = nomes_datasets[int(opcao)]
        clear_output()
        print(f"Dataset '{nome_dataset}' escolhido.")
        try:
            X, y, class_names = carregar_dataset(nome_dataset)
            print(f"Dataset {nome_dataset} carregado com sucesso.")
            print("Classes:", class_names)
            print("Amostras:", X.shape[0], "| Atributos:", X.shape[1])
            break
        except Exception as e:
            print(f"Erro ao processar o dataset: {e}")
    else:
        clear_output()
        print(menu)
        print("Opção inválida. Por favor, escolha um número do menu ou 'Q' para sair.")
        opcao = input("Digite o número do dataset ou 'Q' para sair: ").upper().strip()
