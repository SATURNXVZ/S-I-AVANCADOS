import pickle as pk
import pandas as pd

#o pandas tem um metodo concat()
#criar dataframe 
florNorm = pd.DataFrame(columns = ['sepal_length', 
                'sepal_width', 
                'petal_length', 
                'petal_width',
                'Iris-setosa', 
                'Iris-versicolor', 
                'Iris-virginica'])

#no sisterma final esses dados seráo recebidos
novaFlor = pd.DataFrame([[6.4, 2.8, 5.6, 2.1]], columns = ['sepal_length', 
                                                        'sepal_width', 
                                                        'petal_length', 
                                                        'petal_width',] )

print(novaFlor)
