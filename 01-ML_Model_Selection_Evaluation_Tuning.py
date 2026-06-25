# Importing Libraries:
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_iris
# ---------------------------------------------------------------------
# Loading Data:
data = load_iris()
X = data.data
y = data.target
# ---------------------------------------------------------------------
# Model Building:
model = RandomForestClassifier()
# ---------------------------------------------------------------------
# Defining Hyperparameters:
param_grid = {'n_estimators': [100, 200], 'max_depth': [10,20]}
# ---------------------------------------------------------------------
# Performing Grid Search:
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
grid_search.fit(X, y)
# ---------------------------------------------------------------------
# Showing Result:
print(grid_search.best_params_)
print(grid_search.best_score_)
