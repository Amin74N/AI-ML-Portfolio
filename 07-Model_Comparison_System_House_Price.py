# Importing data:
import pandas as pd
data = {"area":[50,70,90,120,150,170,200,220,250,300],
        "price":[100,150,200,300,400,450,520,600,700,850]}
df = pd.DataFrame(data)
# ---------------------------------------------------------------------
# Target/Feature split:
X = df.drop(columns= ['price'])
y = df['price']
# ---------------------------------------------------------------------
# Train/Test split:
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
# ---------------------------------------------------------------------
# Model building:
from sklearn.linear_model import LinearRegression
LR_model = LinearRegression()
from sklearn.tree import DecisionTreeClassifier
DT_model = DecisionTreeClassifier(random_state=42)
from sklearn.ensemble import RandomForestClassifier
RF_model = RandomForestClassifier(n_estimators=100, random_state=42)
# ---------------------------------------------------------------------
# Model comparison system:
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
def evaluate_model(model, name):
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        print("-------------------------------------")
        print(name)
        print("Train Accuracy Score:", model.score(X_train, y_train))
        print("Test Accuracy Score:", model.score(X_test, y_test))
        print("Mean Absolute Score:", mean_absolute_error(y_test, pred))
        print("R^2 Score:", r2_score(y_test, pred))
evaluate_model(LR_model, ">>> Linear Regression Model")
evaluate_model(DT_model, ">>> Decision Tree Model")
evaluate_model(RF_model, ">>> Random Forest Model")
