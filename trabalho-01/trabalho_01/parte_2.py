# %%

# Parte 2: Regressão com o Wine Quality Dataset

# mypy: disable-error-code="import-untyped"
from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from trabalho_01.utils import print_result

# %%

# Carregue o Wine Quality dataset
wine = datasets.load_wine()

# %%
# Divisão do conjunto de dados

X, y = datasets.make_regression(random_state=0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, train_size=0.7, random_state=0)

# %%
# LinearRegression
lr = LinearRegression()
# treino
lr.fit(X_train, y_train)
# resultado
print_result(lr.score(X_test, y_test), lr.predict(X) == y)

# %%
# RandomForestRegressor

rfr = RandomForestRegressor()
# treino
rfr.fit(X_train, y_train)
# resultado
print_result(rfr.score(X_test, y_test), rfr.predict(X) == y)

# %%
