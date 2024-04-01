# %%
# Parte 3: Classificação com o Digits Dataset
# mypy: disable-error-code="import-untyped"
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from trabalho_01.utils import print_result

# %%
# Carregue o Digits dataset usando sklearn.datasets.load_digits().
datasets.load_digits()
# %%
# Divisão do conjunto de dados

X, y = datasets.make_classification(random_state=0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, train_size=0.7, random_state=0)


# %%
# SVC
svc = SVC()

# treino
svc.fit(X_train, y_train)

# resultado
print_result(svc.score(X_test, y_test), svc.predict(X) == y)

# %%
# KNeighborsClassifier
knc = KNeighborsClassifier()

# treino
knc.fit(X_train, y_train)

# resultado
print_result(knc.score(X_test, y_test), knc.predict(X) == y)
# %%
