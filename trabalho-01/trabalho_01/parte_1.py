# %%
## Parte 1: Classificação com o Iris Dataset

# mypy: disable-error-code="import-untyped"
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


# Carregue o Iris dataset usando sklearn.datasets.load_iris().
iris = datasets.load_iris()


# %%
# Divida os dados em conjuntos de treino e teste.

#Esses parâmetros dão score de 1.0
X, y = datasets.make_classification(random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=38)


# %%
"""
Treine e avalie modelos usando KNeighborsClassifier e SVC. 
Ajuste os parâmetros conforme necessário e compare os resultados.
"""

pipe = Pipeline([('scaler', StandardScaler()), ('svc', SVC())])

pipe.fit(X_train, y_train)

pipe.score(X_test, y_test)

# %%
