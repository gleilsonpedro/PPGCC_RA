import shap
import pandas as pd

def calcular_shap(modelo, X_test):
    # Garantir que X_test seja um DataFrame com as colunas do treino
    X_test = pd.DataFrame(X_test, columns=["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"])

    # Calculando explicações com SHAP (usando KernelExplainer para modelos lineares)
    explainer = shap.KernelExplainer(modelo.predict_proba, X_test)
    shap_values = explainer.shap_values(X_test)
    
    # Exibir as features relevantes (média dos valores SHAP para cada feature)
    print("Importância das features (SHAP):")
    for i, feature_name in enumerate(X_test.columns):
        print(f"{feature_name}: {shap_values[0][:, i].mean():.4f}")
    
    # Gerar o gráfico SHAP
    shap.summary_plot(shap_values, X_test)
