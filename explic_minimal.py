from sklearn import datasets
from sklearn.linear_model import LogisticRegression
import numpy as np

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