from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from z3 import *
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# Carregar o dataset MNIST
mnist = fetch_openml('mnist_784', version=1, cache=True, parser='auto')
X, y = mnist.data, mnist.target
y = y.astype(np.uint8)

# Filtrar as classes 2, 4 e 7
classes_to_keep = [2, 4, 7]
mask = np.isin(y, classes_to_keep)
X_filtered = X[mask]
y_filtered = y[mask]

# Pré-processamento dos dados
scaler = MinMaxScaler()
scaler.fit(X_filtered)
X_filtered = scaler.transform(X_filtered)

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X_filtered, y_filtered, test_size=0.3, stratify=y_filtered)

# Treinar o modelo de Regressão Logística
logistic_regression_classifier = LogisticRegression(multi_class='multinomial', max_iter=200)
logistic_regression_classifier.fit(X_train, y_train)

# Fazer previsões e calcular a precisão
predictions = logistic_regression_classifier.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print("Mnist accuracy:", accuracy)

# Criar variáveis reais para Z3
real_vars = [Real('x_' + str(i)) for i in range(X_filtered.shape[1])] 
domain_constraints = [And(x >= 0, x <= 1) for x in real_vars]

# Definir os termos da regressão logística para cada classe
logistic_reg_terms = [
    logistic_regression_classifier.coef_[i] @ real_vars + logistic_regression_classifier.intercept_[i]
    for i in range(len(classes_to_keep))
]

def minimalExplanation(I, formula):
    """
    Encontrar a explicação minimal para uma imagem.
    
    Args:
        I: Imagem de entrada como uma lista de valores.
        formula: Fórmula Z3 representando a predição correta.
    
    Returns:
        Uma lista de índices de pixels irrelevantes.
    """
    solver = Solver()
    solver.add(formula)
    solver.add(domain_constraints)
    solver.add([real_vars[i] == I[i] for i in range(len(I))])
    
    red_pixels = []
    for i in range(len(I) - 1, -1, -1):
        solver.reset()
        solver.add(formula)
        solver.add(domain_constraints)
        solver.add([real_vars[j] == I[j] for j in range(len(I)) if j != i])
        
        if solver.check() != sat:
            red_pixels.append(i)
    
    return red_pixels

# Encontrar e plotar uma amostra de cada classe com a explicação minimal
for target_class in classes_to_keep:
    # Encontrar uma imagem da classe alvo
    idx = np.where(y_test == target_class)[0][0]
    image = X_test[idx].reshape(28, 28)
    
    # Substituir os valores da imagem nas expressões simbólicas
    term_values = [
        float(simplify(t.subs([(real_vars[i], RealVal(X_test[idx][i])) for i in range(len(X_test[idx]))])).as_decimal())
        if simplify(t.subs([(real_vars[i], RealVal(X_test[idx][i])) for i in range(len(X_test[idx]))])).is_numeral()
        else 0.0  # Retorna 0.0 se a expressão não for um numeral
        for t in logistic_reg_terms
    ]

    # Criar a fórmula Z3 para a classe alvo
    formula = term_values[target_class] > max(
        [term_values[j] for j in range(len(classes_to_keep)) if j != target_class]
    )
    
    # Encontrar a explicação minimal
    red_pixels = minimalExplanation(X_test[idx], formula)

    # Pintar os pixels irrelevantes de vermelho
    img_rgb = Image.fromarray(image).convert("RGB")
    pixels = img_rgb.load()
    vermelho = (255, 0, 0)
    for index in red_pixels:
        x = index % 28
        y = index // 28
        pixels[x, y] = vermelho

    # Plotar a imagem modificada
    plt.imshow(img_rgb)
    plt.title(f"Classe {target_class} - Explicação Minimal")
    plt.axis('off')
    plt.show()