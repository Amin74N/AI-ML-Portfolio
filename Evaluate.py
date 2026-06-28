# Data Loading:
from Preprocessing import X, y, X_train, X_test, y_train, y_test
from Train import LR_model, DT_model, RF_model
# ----------------------------------------------------------------------------------------------------------------------
# Model Evaluation:
from sklearn.metrics import (precision_score, recall_score, f1_score, accuracy_score)
from sklearn.model_selection import cross_val_score
def evaluate_model(model, name):
        pred = model.predict(X_test)
        print("---------------------------------------------------------------------")
        print(name)
        print("Accuracy Score:", accuracy_score(y_test, pred))
        print("Precision Score:", precision_score(y_test, pred))
        print("Recall Score:", recall_score(y_test, pred))
        print("F1 Score:", f1_score(y_test, pred))
        print("Cross Validation Score:", cross_val_score(model,X,y,cv=5).mean())
evaluate_model(LR_model, "Logistic Regression Model:")
evaluate_model(DT_model, "Decision Tree Model:")
evaluate_model(RF_model, "Random Forest Model:")
# ----------------------------------------------------------------------------------------------------------------------
# Best Model Optimization:
from sklearn.model_selection import GridSearchCV
params = {"n_estimators":[50,100], "max_depth":[3,5,10]}
grid = GridSearchCV(RF_model, params, cv=5)
grid.fit(X_train, y_train)
best_model = grid.best_estimator_
import joblib
joblib.dump(RF_model,"Best_RF_Model.pkl")
