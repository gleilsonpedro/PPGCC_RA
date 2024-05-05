from sklearn import datasets
from sklearn.linear_model import LogisticRegression
import numpy as np

# Carregar o conjunto de dados IRIS
iris = datasets.load_iris()
X_iris, y_iris = iris.data, iris.target

# Treinar o modelo de regressão logística, usando o método multinomial
model_iris = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
model_iris.fit(X_iris, y_iris)

# Função para fornecer explicações minimais
def explain_prediction_iris(features, model, target_names):
    probabilities = model.predict_proba([features])[0]
    class_index = np.argmax(probabilities)
    class_probability = probabilities[class_index]
    explanation = f'Classe prevista: {target_names[class_index]} com probabilidade de {class_probability:.2f}\n'
    explanation += 'Coeficientes relevantes para a classe:\n'
    for i, coef in enumerate(model.coef_[class_index]):
        if coef != 0:
            explanation += f'Feature {i}: {coef:.2f}\n'
    return explanation

# Testar o algoritmo com um exemplo do conjunto de dados IRIS
example_index_iris = np.random.choice(len(X_iris))
example_features_iris = X_iris[example_index_iris]
print('Explicação para uma amostra do conjunto de dados Iris:')
print(explain_prediction_iris(example_features_iris, model_iris, iris.target_names))
