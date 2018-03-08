# -*- coding: utf-8 -*-
import pandas as pd
import sklearn.preprocessing as preprocessing
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

print "Carregando o dataset..."

data = pd.read_csv("bank-full.csv", sep=";")

cat_columns = ["job", "marital", "education", "contact", "month", "poutcome"]
num_columns = data.select_dtypes(exclude=['object']).columns
bin_columns = ["default", "housing", "loan", "y"] # yes ou no

aux_columns = cat_columns + bin_columns

# Transforma as strings em ints
# converte objetos em categorias
data[aux_columns] = data[aux_columns].apply(lambda cat: cat.astype('category'))
# aplica os códigos
data[aux_columns] = data[aux_columns].apply(lambda x: x.cat.codes)

# preprocessa as features categóricas como uma matriz de inteiros (one-of-K)
# necessário para usar em modelos lineares
# colunas binárias não são necessárias
# porém comentado pois piora os resultados, talvez sejá necessário reduzir a dimensionalidade
'''
enc = preprocessing.OneHotEncoder()
enc.fit(data[cat_columns])
onehotlabels = enc.transform(data[cat_columns]).toarray()
onehotlabels = pd.DataFrame(onehotlabels)

finaldata = data[num_columns].join(onehotlabels, how='outer').join(data[bin_columns], how='outer')
'''

finaldata = data

features = finaldata.ix[:, :'loan']
label = finaldata['y']

#clf = LinearSVC()
clf = KNeighborsClassifier(n_neighbors=8)
#clf = RandomForestClassifier(criterion='entropy')

print "Treinando um modelo..."
clf.fit(features, label)

scores = cross_val_score(clf, features, label, cv=5)
print("Acurácia: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))