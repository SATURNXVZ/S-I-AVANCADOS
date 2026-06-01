
#TREINAMENTO E COMPARAÇÃO DE CLASSIFICADORES - BANK MARKETING
#modelos: Random Forest | SVM | KNN
# SMOTE aplicado dentro do Pipeline para evitar data leakage

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import RandomizedSearchCV, cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from pickle import dump


#1- CARREGAMENTO E PRÉ-PROCESSAMENTO

dados = pd.read_csv(r"C:\Users\VITORHENRIQUEDEMELO\Documents\S-I-AVANCADOS\EX BANK\bank.csv", sep=";")

print("DATASET: BANK MARKETING")
print(f"Shape: {dados.shape}")
print(f"\nDistribuição das classes (antes do SMOTE):\n{dados['y'].value_counts()}")

#codificar variáveis categóricas
le = LabelEncoder()
colunas_categoricas = dados.select_dtypes(include=["object"]).columns.tolist()
colunas_categoricas.remove("y")

for col in colunas_categoricas:
    dados[col] = le.fit_transform(dados[col])

X = dados.drop(columns=["y"])
y = le.fit_transform(dados["y"])  # no=0, yes=1


#2- HIPERPARAMETRIZAÇÃO - RANDOM FOREST

print("\nHIPERPARAMETRIZAÇÃO - RANDOM FOREST")

rf_grid = {
    "modelo__n_estimators":      [int(x) for x in np.linspace(10, 200, 10)],
    "modelo__criterion":         ["gini", "entropy"],
    "modelo__max_depth":         [int(x) for x in np.linspace(5, 50, 10)],
    "modelo__min_samples_split": [2, 5, 10],
    "modelo__max_features":      ["sqrt", "log2"],
}

rf_pipe = ImbPipeline([
    ("smote",  SMOTE(random_state=42)),
    ("scaler", StandardScaler()),
    ("modelo", RandomForestClassifier(random_state=42)),
])

rf_search = RandomizedSearchCV(
    estimator=rf_pipe,
    param_distributions=rf_grid,
    n_iter=15, cv=5, scoring="accuracy",
    verbose=1, n_jobs=-1, random_state=42,
)
rf_search.fit(X, y)
print(f"Melhores parâmetros RF: {rf_search.best_params_}")


#3- HIPERPARAMETRIZAÇÃO - SVM
print("\nHIPERPARAMETRIZACAO - SVM")

svm_grid = {
    "modelo__C":      [0.1, 1, 10, 100],
    "modelo__kernel": ["rbf", "poly", "sigmoid"],
    "modelo__gamma":  ["scale", "auto"],
}

svm_pipe = ImbPipeline([
    ("smote",  SMOTE(random_state=42)),
    ("scaler", StandardScaler()),
    ("modelo", SVC(random_state=42, probability=True)),
])

svm_search = RandomizedSearchCV(
    estimator=svm_pipe,
    param_distributions=svm_grid,
    n_iter=10, cv=5, scoring="accuracy",
    verbose=1, n_jobs=-1, random_state=42,
)
svm_search.fit(X, y)
print(f"Melhores parâmetros SVM: {svm_search.best_params_}")


#4- HIPERPARAMETRIZACAO - KNN

print("\nHIPERPARAMETRIZAÇÃO - KNN")

knn_grid = {
    "modelo__n_neighbors": list(range(3, 21, 2)),
    "modelo__weights":     ["uniform", "distance"],
    "modelo__metric":      ["euclidean", "manhattan", "minkowski"],
}

knn_pipe = ImbPipeline([
    ("smote",  SMOTE(random_state=42)),
    ("scaler", StandardScaler()),
    ("modelo", KNeighborsClassifier()),
])

knn_search = RandomizedSearchCV(
    estimator=knn_pipe,
    param_distributions=knn_grid,
    n_iter=10, cv=5, scoring="accuracy",
    verbose=1, n_jobs=-1, random_state=42,
)
knn_search.fit(X, y)
print(f"Melhores parâmetros KNN: {knn_search.best_params_}")


# 5- AVALIAÇÃO COM CROSS-VALIDATION

print("\nAVALIAÇÃO COM CROSS-VALIDATION (10-fold)")

rf_params  = {k.replace("modelo__", ""): v for k, v in rf_search.best_params_.items()}
svm_params = {k.replace("modelo__", ""): v for k, v in svm_search.best_params_.items()}
knn_params = {k.replace("modelo__", ""): v for k, v in knn_search.best_params_.items()}

pipelines = {
    "Random Forest": ImbPipeline([
        ("smote",  SMOTE(random_state=42)),
        ("scaler", StandardScaler()),
        ("modelo", RandomForestClassifier(**rf_params, random_state=42)),
    ]),
    "SVM": ImbPipeline([
        ("smote",  SMOTE(random_state=42)),
        ("scaler", StandardScaler()),
        ("modelo", SVC(**svm_params, probability=True, random_state=42)),
    ]),
    "KNN": ImbPipeline([
        ("smote",  SMOTE(random_state=42)),
        ("scaler", StandardScaler()),
        ("modelo", KNeighborsClassifier(**knn_params)),
    ]),
}

metricas_cv = ["precision_macro", "recall_macro", "f1_macro", "accuracy"]
resultados  = {}

for nome, pipeline in pipelines.items():
    scores = cross_validate(
        pipeline,
        X, y,
        scoring=metricas_cv,
        cv=10,
        n_jobs=-1,
    )
    resultados[nome] = {
        "Acurácia":       scores["test_accuracy"].mean(),
        "Precisão":       scores["test_precision_macro"].mean(),
        "Recall":         scores["test_recall_macro"].mean(),
        "F1-Score":       scores["test_f1_macro"].mean(),
        "Acurácia ± std": scores["test_accuracy"].std(),
    }

print("\n{:<20} {:>10} {:>10} {:>10} {:>10} {:>12}".format(
    "Modelo", "Acurácia", "Precisão", "Recall", "F1-Score", "Std (Acc)"
))
print("-" * 72)
for nome, r in resultados.items():
    print("{:<20} {:>9.2%} {:>9.2%} {:>9.2%} {:>9.2%} {:>11.4f}".format(
        nome, r["Acurácia"], r["Precisão"], r["Recall"], r["F1-Score"], r["Acurácia ± std"],
    ))


#6- GRÁFICO COMPARATIVO

nomes     = list(resultados.keys())
acuracias = [r["Acurácia"] for r in resultados.values()]
stds      = [r["Acurácia ± std"] for r in resultados.values()]
cores     = ["#4C72B0", "#DD8452", "#55A868"]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Comparação de Classificadores – Bank Marketing", fontsize=14, fontweight="bold")

bars = axes[0].bar(nomes, acuracias, yerr=stds, color=cores, capsize=6, edgecolor="black")
axes[0].set_ylim(0.6, 1.0)
axes[0].set_ylabel("Acurácia Média (CV 10-fold)")
axes[0].set_title("Acurácia com Desvio Padrão")
for bar, val in zip(bars, acuracias):
    axes[0].text(bar.get_x() + bar.get_width() / 2, val + 0.005,
                 f"{val:.2%}", ha="center", va="bottom", fontsize=10)

metricas_plot = ["Acurácia", "Precisão", "Recall", "F1-Score"]
x     = np.arange(len(metricas_plot))
width = 0.25

for i, (nome, r) in enumerate(resultados.items()):
    vals = [r[m] for m in metricas_plot]
    axes[1].bar(x + i * width, vals, width, label=nome, color=cores[i], edgecolor="black")

axes[1].set_xticks(x + width)
axes[1].set_xticklabels(metricas_plot)
axes[1].set_ylim(0.6, 1.0)
axes[1].set_title("Métricas Comparativas")
axes[1].set_ylabel("Valor Médio")
axes[1].legend()

plt.tight_layout()
plt.savefig("comparacao_modelos_bank.png", dpi=150)
plt.show()


#7- MELHOR MODELO → TREINAMENTO FINAL + SALVAR

melhor_nome     = max(resultados, key=lambda n: resultados[n]["F1-Score"])
melhor_pipeline = pipelines[melhor_nome]

print(f"\n=======================")
print(f"MELHOR MODELO: {melhor_nome}")
print(f"F1-Score médio: {resultados[melhor_nome]['F1-Score']:.2%}")

melhor_pipeline.fit(X, y)

#salvar o pipeline inteiro (já inclui SMOTE + scaler + modelo)
dump(melhor_pipeline, open("bank_modelo.pkl", "wb"))
print("\nPipeline salvo em: bank_modelo.pkl")


#8- EXEMPLO DE INFERÊNCIA

print("\nExemplo de inferência")
nova_instancia = pd.DataFrame([X.iloc[0].values], columns=X.columns)

predicao      = melhor_pipeline.predict(nova_instancia)
probabilidade = melhor_pipeline.predict_proba(nova_instancia)

print(f"Classes:        {melhor_pipeline.classes_}  (0=no, 1=yes)")
print(f"Predição:       {predicao[0]}  ({'Assinou' if predicao[0] == 1 else 'Não assinou'})")
print(f"Probabilidades: {probabilidade[0]}")