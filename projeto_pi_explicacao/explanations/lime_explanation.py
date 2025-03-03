from lime.lime_tabular import LimeTabularExplainer

def calcular_lime(modelo, X_train, X_test):
    # Criar o explicador LIME
    explainer_lime = LimeTabularExplainer(X_train.values, mode='classification', training_labels=y_train, feature_names=X_train.columns)

    # Explicar uma instância do teste
    explanation = explainer_lime.explain_instance(X_test.iloc[0].values, modelo.predict_proba)
    
    # Exibir as features mais importantes
    print("Features relevantes (LIME):")
    for feature, weight in explanation.as_list():
        print(f"{feature}: {weight:.4f}")

    # Mostrar o gráfico do LIME
    explanation.show_in_notebook()
