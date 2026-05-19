#treinamento de classificadores

from sklearn.model_selection import train_test_split
import pandas as pd
#precisamos de um metaestimador
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE



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

fertilityTree = tree.fit(atributoTrain, classeTrain)

#pre teste
predicts = fertilityTree.predict(atributoTest)

#ConfusionMatrixDisplay.from_estimator(fertilityTree, atributoTest, classeTest)

plt.show()

#acuracia geral 
acuracia = accuracy_score(classeTest, predicts)
print('acuracia: ', acuracia)

vn, fn, vp, fp = confusion_matrix(classeTest, predicts).ravel()

#especificidade = vn/(vn+fp)
especify = vn/(vn+vp)

#sensibilidade = vp/(vp+fn)
sensibility = vp/(vp+fn)

print('Especifidade: ', especify)
print('Sensibility: ', sensibility)

