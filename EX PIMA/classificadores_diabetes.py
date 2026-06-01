#TREINAMENTO E COMPARAÇÃO DE CLASSIFICADORES - PIMA INDIANS DIABETES
#modelos: Random Forest | SVM | KNN
#=============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, confusion_matrix, ConfusionMatrixDisplay,
    classification_report
)
from imblearn.over_sampling import SMOTE
from pickle import dump


#CARREGAMENTO E PRÉ-PROCESSAMENTO
dados = pd.read_csv(r"C:\Users\VITORHENRIQUEDEMELO\Documents\S-I-AVANCADOS\EX PIMA\diabetes.csv")


print("DATASET: PIMA INDIANS DIABETES")
print(f"Shape: {dados.shape}")
print(f"\nDistribuição das classes (antes do SMOTE):\n{dados['Outcome'].value_counts()}")

X = dados.drop(columns=["Outcome"])
y = dados["Outcome"]

#balancear classes com SMOTE
smote = SMOTE(random_state=42)
X_bal, y_bal = smote.fit_resample(X, y)

print(f"\nDistribuição após SMOTE:\n{y_bal.value_counts()}")

#normalizar os dados (necessário especialmente para SVM e KNN)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_bal)

# =============================================================================
# 2. HIPERPARAMETRIZAÇÃO - RANDOM FOREST
# =============================================================================

print("\n" + "=" * 60)
print("HIPERPARAMETRIZAÇÃO - RANDOM FOREST")
print("=" * 60)

rf_grid = {
    "n_estimators": [int(x) for x in np.linspace(10, 200, 10)],
    "criterion": ["gini", "entropy"],
    "max_depth": [int(x) for x in np.linspace(5, 50, 10)],
    "min_samples_split": [2, 5, 10],
    "max_features": ["sqrt", "log2"],
}

rf_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_distributions=rf_grid,
    n_iter=15,
    cv=5,
    scoring="accuracy",
    verbose=1,
    n_jobs=-1,
    random_state=42,
)
rf_search.fit(X_scaled, y_bal)
print(f"Melhores parâmetros RF: {rf_search.best_params_}")

# =============================================================================
# 3. HIPERPARAMETRIZAÇÃO - SVM
# =============================================================================

print("\n" + "=" * 60)
print("HIPERPARAMETRIZAÇÃO - SVM")
print("=" * 60)

svm_grid = {
    "C": [0.1, 1, 10, 100],
    "kernel": ["rbf", "poly", "sigmoid"],
    "gamma": ["scale", "auto"],
}

svm_search = RandomizedSearchCV(
    estimator=SVC(random_state=42, probability=True),
    param_distributions=svm_grid,
    n_iter=10,
    cv=5,
    scoring="accuracy",
    verbose=1,
    n_jobs=-1,
    random_state=42,
)
svm_search.fit(X_scaled, y_bal)
print(f"Melhores parâmetros SVM: {svm_search.best_params_}")

# =============================================================================
# 4. HIPERPARAMETRIZAÇÃO - KNN
# =============================================================================

print("\n" + "=" * 60)
print("HIPERPARAMETRIZAÇÃO - KNN")
print("=" * 60)

knn_grid = {
    "n_neighbors": list(range(3, 21, 2)),
    "weights": ["uniform", "distance"],
    "metric": ["euclidean", "manhattan", "minkowski"],
}

knn_search = RandomizedSearchCV(
    estimator=KNeighborsClassifier(),
    param_distributions=knn_grid,
    n_iter=10,
    cv=5,
    scoring="accuracy",
    verbose=1,
    n_jobs=-1,
    random_state=42,
)
knn_search.fit(X_scaled, y_bal)
print(f"Melhores parâmetros KNN: {knn_search.best_params_}")

# =============================================================================
# 5. TREINAMENTO FINAL COM CROSS-VALIDATION (10-fold)
# =============================================================================

print("\n" + "=" * 60)
print("AVALIAÇÃO COM CROSS-VALIDATION (10-fold)")
print("=" * 60)

modelos = {
    "Random Forest": RandomForestClassifier(**rf_search.best_params_, random_state=42),
    "SVM":           SVC(**svm_search.best_params_, probability=True, random_state=42),
    "KNN":           KNeighborsClassifier(**knn_search.best_params_),
}

metricas_cv = ["precision_macro", "recall_macro", "f1_macro", "accuracy"]
resultados = {}

for nome, modelo in modelos.items():
    scores = cross_validate(
        modelo,
        X_scaled,
        y_bal,
        scoring=metricas_cv,
        cv=10,
        n_jobs=-1,
    )
    resultados[nome] = {
        "Acurácia":  scores["test_accuracy"].mean(),
        "Precisão":  scores["test_precision_macro"].mean(),
        "Recall":    scores["test_recall_macro"].mean(),
        "F1-Score":  scores["test_f1_macro"].mean(),
        "Acurácia ± std": scores["test_accuracy"].std(),
    }

# Exibir tabela comparativa
print("\n{:<20} {:>10} {:>10} {:>10} {:>10} {:>12}".format(
    "Modelo", "Acurácia", "Precisão", "Recall", "F1-Score", "Std (Acc)"
))
print("-" * 72)
for nome, r in resultados.items():
    print("{:<20} {:>9.2%} {:>9.2%} {:>9.2%} {:>9.2%} {:>11.4f}".format(
        nome,
        r["Acurácia"],
        r["Precisão"],
        r["Recall"],
        r["F1-Score"],
        r["Acurácia ± std"],
    ))

# =============================================================================
# 6. GRÁFICO COMPARATIVO
# =============================================================================

nomes = list(resultados.keys())
acuracias = [r["Acurácia"] for r in resultados.values()]
stds      = [r["Acurácia ± std"] for r in resultados.values()]
cores     = ["#4C72B0", "#DD8452", "#55A868"]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Comparação de Classificadores – Pima Indians Diabetes", fontsize=14, fontweight="bold")

# --- Barras de acurácia ---
bars = axes[0].bar(nomes, acuracias, yerr=stds, color=cores, capsize=6, edgecolor="black")
axes[0].set_ylim(0.7, 1.0)
axes[0].set_ylabel("Acurácia Média (CV 10-fold)")
axes[0].set_title("Acurácia com Desvio Padrão")
for bar, val in zip(bars, acuracias):
    axes[0].text(bar.get_x() + bar.get_width() / 2, val + 0.005,
                 f"{val:.2%}", ha="center", va="bottom", fontsize=10)

# --- Radar / barras de todas as métricas ---
metricas_plot = ["Acurácia", "Precisão", "Recall", "F1-Score"]
x = np.arange(len(metricas_plot))
width = 0.25

for i, (nome, r) in enumerate(resultados.items()):
    vals = [r[m] for m in metricas_plot]
    axes[1].bar(x + i * width, vals, width, label=nome, color=cores[i], edgecolor="black")

axes[1].set_xticks(x + width)
axes[1].set_xticklabels(metricas_plot)
axes[1].set_ylim(0.7, 1.0)
axes[1].set_title("Métricas Comparativas")
axes[1].set_ylabel("Valor Médio")
axes[1].legend()

plt.tight_layout()
plt.savefig("comparacao_modelos.png", dpi=150)
plt.show()

# =============================================================================
# 7. MELHOR MODELO → TREINAMENTO FINAL + SALVAR
# =============================================================================

melhor_nome = max(resultados, key=lambda n: resultados[n]["F1-Score"])
melhor_modelo = modelos[melhor_nome]

print(f"\n{'=' * 60}")
print(f"MELHOR MODELO: {melhor_nome}")
print(f"F1-Score médio: {resultados[melhor_nome]['F1-Score']:.2%}")
print(f"{'=' * 60}")

# Treinar no conjunto completo (sem holdout, pois já validamos via CV)
melhor_modelo.fit(X_scaled, y_bal)

# Salvar modelo e scaler
dump(melhor_modelo, open("diabetes_modelo.pkl", "wb"))
dump(scaler,        open("diabetes_scaler.pkl", "wb"))

print("\nModelo salvo em: diabetes_modelo.pkl")
print("Scaler salvo em: diabetes_scaler.pkl")

# =============================================================================
# 8. EXEMPLO DE INFERÊNCIA
# =============================================================================

print("\n--- Exemplo de inferência ---")
nova_instancia = pd.DataFrame([[6, 148, 72, 35, 0, 33.6, 0.627, 50]], columns=X.columns)
nova_instancia_scaled = scaler.transform(nova_instancia)

predicao    = melhor_modelo.predict(nova_instancia_scaled)
probabilidade = melhor_modelo.predict_proba(nova_instancia_scaled)

print(f"Classes:      {melhor_modelo.classes_}")
print(f"Predição:     {predicao[0]}  ({'Diabético' if predicao[0] == 1 else 'Não diabético'})")
print(f"Probabilidades: {probabilidade[0]}")
