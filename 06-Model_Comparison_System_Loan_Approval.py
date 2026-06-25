# Importing data:
import pandas as pd
df = pd.read_csv('loan.csv')
# ---------------------------------------------------------------------
# Target/Feature split:
X = df.drop(columns= ['approved'])
y = df['approved']
# ---------------------------------------------------------------------
# Train/Test split:
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
# ---------------------------------------------------------------------
# Model building:
from sklearn.linear_model import LogisticRegression
LR_model = LogisticRegression(max_iter=1000)
from sklearn.tree import DecisionTreeClassifier
DT_model = DecisionTreeClassifier(random_state=42)
from sklearn.ensemble import RandomForestClassifier
RF_model = RandomForestClassifier(n_estimators=100, random_state=42)
# ---------------------------------------------------------------------
# Grid search for random forrest model:
from sklearn.model_selection import GridSearchCV
params = {"n_estimators": [50,100,200], "max_depth": [3, 5, 10]}
grid = GridSearchCV(RF_model, params, cv=5)
grid.fit(X_train, y_train)
RF_model = grid.best_estimator_
# ---------------------------------------------------------------------
# Saving the best random forrest model:
import joblib
joblib.dump(RF_model,"Best_RF_Model.pkl")
# ---------------------------------------------------------------------
# Model comparison system:
from sklearn.metrics import (precision_score,recall_score,f1_score)
from sklearn.model_selection import cross_val_score
def evaluate_model(model, name):
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        print(name)
        print("Train Accuracy Score:", model.score(X_train, y_train))
        print("Test Accuracy Score:", model.score(X_test, y_test))
        print("Precision Score:", precision_score(y_test, pred, average='weighted'))
        print("Recall Score:", recall_score(y_test, pred, average='weighted'))
        print("F1 Score:", f1_score(y_test, pred, average='weighted'))
        print("Cross Validation Score:", cross_val_score(model,X,y,cv=5).mean())
        print("-------------------------------------")
evaluate_model(LR_model, ">>> Logistic Regression Model")
evaluate_model(DT_model, ">>> Decision Tree Model")
evaluate_model(RF_model, ">>> Random Forest Model")
print("Best setting for RandomForest model:\n", grid.best_params_)
