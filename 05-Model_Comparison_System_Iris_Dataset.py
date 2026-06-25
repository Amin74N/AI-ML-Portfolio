# Importing data:
from sklearn.datasets import load_iris
iris = load_iris()
# ---------------------------------------------------------------------
# Target/Feature split:
X = iris.data
y = iris.target
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
# Model comparison system:
from sklearn.metrics import (precision_score,recall_score,f1_score)
from sklearn.model_selection import cross_val_score
def evaluate_model(model, name):
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        print("-----------------------------------------------\n", name)
        print("Train Accuracy Score:", model.score(X_train, y_train))
        print("Test Accuracy Score:", model.score(X_test, y_test))
        print("Precision Score:", precision_score(y_test, pred, average='weighted'))
        print("Recall Score:", recall_score(y_test, pred, average='weighted'))
        print("F1 Score:", f1_score(y_test, pred, average='weighted'))
        print("Cross Validation Score:", cross_val_score(model,X,y,cv=5).mean())
evaluate_model(LR_model, "Logistic Regression Model:")
evaluate_model(DT_model, "Decision Tree Model:")
evaluate_model(RF_model, "Random Forest Model:")
