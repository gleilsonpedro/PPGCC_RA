from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd
from datasets import 

# Função para transformar o dataset em um problema binário (classe 0 contra as outras)
def transformar_problema_binario(y, classe_0):
    return [1 if label == classe_0 else 0 for label in y]

def analisar_instancias(X, y, class_names, classe_0=0, instancia_para_analisar=None):
    global TUDO
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])

    # Transforma o problema em binário
    y_binario = transformar_problema_binario(y, classe_0)

    # Divide o dataset em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y_binario, test_size=0.2, random_state=42)
    
    # Treina o modelo
    modelo = LogisticRegression(max_iter=200)
    modelo.fit(X_train, y_train)

    # Obtém os nomes das features
    feature_names = X.columns.tolist()  
    
    # Seleciona as instâncias para análise
    num_instancias = len(X_test)
    instancias_para_analisar = range(num_instancias) if instancia_para_analisar is None else [instancia_para_analisar]
    
    TUDO = []
    # Loop para analisar cada instância selecionada
    for idx in instancias_para_analisar:
        Vs = X_test.iloc[idx].to_dict()
        instancia_test = X_test.iloc[[idx]]

        # Calcula `gamma_A` usando `decision_function`
        gamma_A = modelo.decision_function(instancia_test)[0]
        
        # Cálculo do valor delta para cada feature
        delta = []
        w = modelo.coef_[0]
        for i, feature in enumerate(feature_names):
            if w[i] < 0:
                delta.append((Vs[feature] - X[feature].max()) * w[i])
            else:
                delta.append((Vs[feature] - X[feature].min()) * w[i])

        # Calcula R
        R = sum(delta) - gamma_A
        
        # Computa a PI-explicação para a instância atual usando nomes das features
        Xpl = one_explanation(Vs, delta, R, feature_names, modelo, instancia_test, X)
        # Imprime os resultados
        classe_verdadeira = y_test[idx]
        print(f"\nInstância {idx}:")
        print(f"Classe verdadeira (binária): {classe_verdadeira}")
        print(f"PI-Explicação: ")
        
        TUDO.append(Xpl)

        for item in Xpl:
            print(f"- {item}")
        
        
        if not Xpl:
            print('_No-PI-explanation_'*3)
        

# Função para calcular a PI-explicação e incluir os intervalos de valores mínimos e máximos que garantem a classe
def one_explanation(Vs, delta, R, feature_names, modelo, instancia_test, X):
    limiar_delta = np.percentile(np.abs(delta), 25)  # Pega o percentil 25 dos deltas
    Xpl = []
    delta_sorted = sorted(enumerate(delta), key=lambda x: abs(x[1]), reverse=True)
    R_atual = R
    Idx = 0
    
    while R_atual >= 0 and Idx < len(delta_sorted):
        sorted_idx, delta_value = delta_sorted[Idx]
        feature = feature_names[sorted_idx]
        feature_value = Vs[feature]

        if abs(delta_value) < limiar_delta:  # Descarta deltas muito pequenos
           break

        # Encontra os valores mínimo e máximo para manter a classe usando perturbação
        #min_val, max_val = encontrar_intervalo_perturbacao(modelo, instancia_test, feature, feature_value, classe_desejada=1, X=X)

        # Adiciona a feature com o valor da instância e o intervalo mínimo/máximo que mantém a classe
        Xpl.append(f"{feature} - {feature_value} ")
        R_atual -= delta_value
        Idx += 1
    
    return Xpl

# Função para encontrar o intervalo de perturbação para manter a classe, considerando limites de X
def encontrar_intervalo_perturbacao(modelo, instancia, feature, valor_original, classe_desejada, X, passo=0.1, max_iter=50):
    # Define os valores mínimo e máximo baseados nos dados de entrada
    min_val_data = X[feature].min()
    max_val_data = X[feature].max()
    
    # Inicializa os valores mínimo e máximo com o valor da instância
    min_val, max_val = valor_original, valor_original
    
    # Perturba negativamente
    for _ in range(max_iter):
        min_val -= passo
        if min_val < min_val_data:
            min_val = min_val_data
            break
        instancia_perturbada = instancia.copy()
        instancia_perturbada[feature] = min_val
        predicao = modelo.predict(instancia_perturbada)
        if predicao[0] != classe_desejada:
            min_val += passo
            break

    # Perturba positivamente
    for _ in range(max_iter):
        max_val += passo
        if max_val > max_val_data:
            max_val = max_val_data
            break
        instancia_perturbada = instancia.copy()
        instancia_perturbada[feature] = max_val
        predicao = modelo.predict(instancia_perturbada)
        if predicao[0] != classe_desejada:
            max_val -= passo
            break

    return min_val, max_val

# Exemplo de uso (substitua X e y pelos dados adequados)
analisar_instancias(X, y, class_names, classe_0=0)
# Crie um dicionário para armazenar a contagem de cada feature
contagem_features = {}

# Itere sobre cada item da lista TUDO
for item in TUDO:
    # Verifique se o item é uma lista
    if isinstance(item, list):
        # Itere sobre cada item da lista
        for feature in item:
            # Extraia o nome da feature
            nome_feature = feature.split(" - ")[0]

            # Verifique se a feature já está no dicionário
            if nome_feature in contagem_features:
                # Incremente a contagem
                contagem_features[nome_feature] += 1
            else:
                # Adicione a feature ao dicionário com contagem 1
                contagem_features[nome_feature] = 1

# Imprima quantidade em que as features aparecem na explicacão em ordem
for nome_feature, contagem in contagem_features.items():
    print(f"Feature: {nome_feature} Contagem: {contagem}")