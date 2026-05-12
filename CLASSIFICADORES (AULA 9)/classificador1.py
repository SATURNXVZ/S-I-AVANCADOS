#treinamento de classificadores

from sklearn.model_selection import train_test_split
import pandas as pd
#precisamos de um metaestimador
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt



dados =pd.read_csv("fertility_Diagnosis.bolnho")

dadosAtributos = dados.drop(columns=['Diagnostico'])

dadosClasses = dados['Diagnostico']

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