# %%
## Parte 1: Classificação com o Iris Dataset

# mypy: disable-error-code="import-untyped"
import numpy
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# %%
# Carregar o dataset
iris = datasets.load_iris()

# %%
# Divisão do conjunto de dados

X, y = datasets.make_classification( random_state=0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, train_size=0.7, random_state=0)

# %%
# SVC
svc = SVC()

# treino
svc.fit(X_train, y_train)

# resultado
svc_score = svc.score(X_test, y_test)
print('Score: ', svc_score)

svc_result: numpy.ndarray = svc.predict(X) == y

svc_unique, svc_count = numpy.unique(svc_result, return_counts=True)
svc_count_true = dict(zip(svc_unique, svc_count))[True]
print(f'Predict: {svc_count_true} / {len(svc_result)} ' )




# %%
# KNeighborsClassifier
knc = KNeighborsClassifier()

# treino
knc.fit(X_train, y_train)

# resultado
knc_score = knc.score(X_test, y_test)
print('Score: ', knc_score)

knc_result: numpy.ndarray = knc.predict(X) == y

knc_unique, knc_count = numpy.unique(knc_result, return_counts=True)
knc_count_true = dict(zip(knc_unique, knc_count))[True]
print(f'Predict: {knc_count_true} / {len(knc_result)} ' )
# %%
