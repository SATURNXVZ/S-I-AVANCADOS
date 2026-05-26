from pickle import load

#abrir o modelo
fertility_model = load(open('fertility_rf.pkl', 'rb'))

#coletar dados de nova instancia
nova_instancia = [0.33,0.78,1,0,0,0,1,1,0.06]
result = fertility_model.predict_proba([nova_instancia])

print(fertility_model.classes_)
print(result)

#treinar bank marketing (45 mil linhas - 16 linhas)
#não normalizar target
#fazer com random forest

#PIMA INDIANS DIABETES
#rodar random forest, um estimador;
#testar vários outros estimadores para perceber a diferença
#fazer grid search e anotar as acurácias para ver qual delas é a melhor.
#após, usar a com melhor acuracia.

