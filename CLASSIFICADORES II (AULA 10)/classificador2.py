#treinamento de classificadores

from sklearn.model_selection import train_test_split
import pandas as pd
#precisamos de um metaestimador
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from pickle import dump
import numpy as np
from sklearn.model_selection import RandomizedSearchCV


dados =pd.read_csv("CLASSIFICADORES (AULA 9)/fertility_Diagnosis.bolnho")

#separar atributos e classes
dadosAtributos = dados.drop(columns=['Diagnostico'])
dadosClasses = dados['Diagnostico']

#balancear as classes
balancer = SMOTE()
dadosAtributos, dadosClasses = balancer.fit_resample(dadosAtributos, dadosClasses)

#frequencia de classes originais
print('Frequencia das classes balanceadas: ', dadosClasses.value_counts)

#Segmentar ps dadps em dados para treinamento
atributoTrain, atributoTest, classeTrain, classeTest = train_test_split(dadosAtributos, dadosClasses, test_size=0.3)

#treinar modelo
tree = DecisionTreeClassifier(random_state=42)

#hiperparametrização da Random Forest
# definir os dominios para os hiperparametros
nEstimators = [int(x) for x in np.linspace(start=10, stop=100, num=10)]
criterion = ['gini', 'entrpy'] #log loss ainda não

min_samples_split = [int(x) for x in np.linspace(start=2, stop=10, num=2)]
max_depth = [int(x) for x in np.linspace(start=10, stop=100, num=20)]

max_features = ['sqrt', 'log2']

#criar grade de valores
RF_grid = {
    'nEstimators': nEstimators,
    'criterion' : criterion,
    'min_samples_split' : min_samples_split,
    'max_depth' : max_depth,
    'max_features' : max_features
    }

rf = RandomForestClassifier(random_state=42)
Rf_Hyper = RandomizedSearchCV(
    estimator=rf, 
    param_distributions=RF_grid,
    n_iter=10,
    cv=3,
    verbose=2,
    n_jobs=-1
)

Rf_Hyper.fit(dadosAtributos, dadosClasses)

fertilityTree = tree.fit(atributoTrain, classeTrain)
fertilityRF = rf.fit(atributoTrain, classeTrain)

from pprint import pprint
pprint(Rf_Hyper.best_params_)

# #pre teste
# predicts = fertilityRF.predict(atributoTest)

# #salvar o modelo
# dump(fertilityTree, open('fertility_rf.pkl', 'wb'))
# #ConfusionMatrixDisplay.from_estimator(fertilityTree, atributoTest, classeTest)

# dump(fertilityRF, open('fertility_tree.pkl', 'wb'))

# plt.show()

# #acuracia geral 
# acuracia = accuracy_score(classeTest, predicts)
# print('acuracia: ', acuracia)

# vn, fn, vp, fp = confusion_matrix(classeTest, predicts).ravel()

# #especificidade = vn/(vn+fp)
# especify = vn/(vn+vp)

# #sensibilidade = vp/(vp+fn)
# sensibility = vp/(vp+fn)

# print('Especifidade: ', especify)
# print('Sensibility: ', sensibility)

