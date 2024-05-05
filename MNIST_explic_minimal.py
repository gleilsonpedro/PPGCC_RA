from sklearn import datasets
from sklearn.linear_model import LogisticRegression
import numpy as np
from IRIS_explic_minimal import *

# Carregar o conjunto de dados MNIST
mnist = datasets.fetch_openml('mnist_784', version=1)
X_mnist, y_mnist = mnist.data, mnist.target.astype(int)

# Filtrar para as classes 2, 4 e 7
mnist_classes = [2, 4, 7]
indices = np.isin(y_mnist, mnist_classes)
X_mnist_filtered = X_mnist[indices]
y_mnist_filtered = y_mnist[indices]

# Mapear as classes para um novo conjunto de 0, 1, 2
class_mapping = {2: 0, 4: 1, 7: 2}
y_mnist_mapped = np.vectorize(class_mapping.get)(y_mnist_filtered)

# Treinar o modelo de regressão logística
model_mnist = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
model_mnist.fit(X_mnist_filtered, y_mnist_mapped)

# Função para fornecer explicações minimais
def explain_prediction_mnist(features, model, target_names):
    probabilities = model.predict_proba([features])[0]
    class_index = np.argmax(probabilities)
    class_probability = probabilities[class_index]
    explanation = f'Classe prevista: {target_names[class_index]} com probabilidade de {class_probability:.2f}\n'
    explanation += 'Coeficientes relevantes para a classe:\n'
    for i, coef in enumerate(model.coef_[class_index]):
        if coef != 0:
            explanation += f'Feature {i}: {coef:.2f}\n'
    return explanation

# Testar o algoritmo com um exemplo do conjunto de dados MNIST
example_index_mnist = np.random.choice(y_mnist_mapped.shape[0])
example_features_mnist = X_mnist_filtered.to_numpy()[example_index_mnist]
print('Explicação para uma amostra do conjunto de dados MNIST:')
print(explain_prediction_mnist(example_features_mnist, model_mnist, ['Classe 2', 'Classe 4', 'Classe 7']))

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.linear_model import LogisticRegression

# Função para calcular a explicação mínima
def minimal_explanation(model, X):
    explanations = []
    for i in range(len(X)):
        probabilities = model.predict_proba([X[i]])[0]
        class_index = np.argmax(probabilities)
        relevant_features = np.where(model.coef_[class_index] != 0)[0]
        explanations.append(relevant_features)
    return explanations

# Função para plotar a explicação de uma instância do MNIST
def plot_explanation(instance_index, explanation, X):
    instance = X[instance_index].reshape(28, 28)
    explanation_indices = explanation[instance_index]
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(instance, cmap='gray')
    ax[0].set_title('Instância Original')
    ax[0].axis('off')
    explanation_image = np.zeros_like(instance)
    explanation_image[explanation_indices % 28, explanation_indices // 28] = 1
    ax[1].imshow(instance, cmap='gray')
    ax[1].imshow(explanation_image, cmap='Reds', alpha=0.5)
    ax[1].set_title('Explicação Minimal')
    ax[1].axis('off')
    plt.show()

# Carregar o conjunto de dados MNIST
mnist = datasets.fetch_openml('mnist_784', version=1)
X_mnist, y_mnist = mnist.data.astype(float), mnist.target.astype(int)

# Normalizar os valores de pixel para o intervalo [0, 1]
X_mnist /= 255.0

# Filtrar para as classes 2, 4 e 7
mnist_classes = [2, 4, 7]
indices = np.isin(y_mnist, mnist_classes)
X_mnist_filtered = X_mnist[indices]
y_mnist_filtered = y_mnist[indices]

# Treinar o modelo de regressão logística
model_mnist = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
model_mnist.fit(X_mnist_filtered, y_mnist_filtered)

# Converter X_mnist_filtered em uma matriz numpy
X_mnist_filtered = X_mnist_filtered.to_numpy()

# Calcular a explicação minimal
explanation_mnist = minimal_explanation(model_mnist, X_mnist_filtered)

# Escolha uma instância para plotar a explicação
instance_index = np.random.choice(len(X_mnist_filtered))

# Plotar a explicação
plot_explanation(instance_index, explanation_mnist, X_mnist_filtered)
