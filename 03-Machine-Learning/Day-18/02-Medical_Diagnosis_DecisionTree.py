# Importing data:
import pandas as pd
data = pd.read_csv('medical.csv')
# ---------------------------------------------------------------------
# Target/Feature split:
X = data.drop(columns= ['disease'])
y = data['disease']
# ---------------------------------------------------------------------
# Train/Test split:
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# ---------------------------------------------------------------------
# Model Building:
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(random_state=42)
# ---------------------------------------------------------------------
# Grid Search:
from sklearn.model_selection import GridSearchCV
params = {"max_depth": [2, 3, 5, 10]}
grid = GridSearchCV(model, params, cv=5)
grid.fit(X_train, y_train)
print("Best Setting:", grid.best_params_)
# ---------------------------------------------------------------------
# Model training:
bestmodel = grid.best_estimator_
# ---------------------------------------------------------------------
# Calculating Train/Test Accuracy Score:
train_score = bestmodel.score(X_train, y_train)
test_score = bestmodel.score(X_test, y_test)
# ---------------------------------------------------------------------
# Showing Train/Test Accuracy Scores:
print("Train Accuracy Score:", train_score)
print("Test Accuracy Score:", test_score)
# ---------------------------------------------------------------------
# Loading Classification Criteria:
from sklearn.metrics import (precision_score,recall_score,f1_score)
pred = bestmodel.predict(X_test)
# ---------------------------------------------------------------------
# Showing Classification Criteria Scores:
print("Precision Score:", precision_score(y_test,pred))
print("Recall Score:", recall_score(y_test,pred))
print("F1 Score:", f1_score(y_test,pred))
