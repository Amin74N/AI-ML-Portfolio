# Importing data:
import pandas as pd
df = pd.read_csv('employee.csv')
#---------------------------------------------------------------------
# Showing raw data:
print(df)
#---------------------------------------------------------------------
# Showing EDA:
print("Brief EDA:\n", df.describe())
#---------------------------------------------------------------------
# Target/Feature split:
X = df.drop(columns= ['promoted'])
y = df['promoted']
#---------------------------------------------------------------------
# Train/Test split:
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
#---------------------------------------------------------------------
# Model building:
from sklearn.tree import DecisionTreeClassifier
tree_model = DecisionTreeClassifier(random_state=42)
from sklearn.ensemble import RandomForestClassifier
forrest_model = RandomForestClassifier(n_estimators=100)
#---------------------------------------------------------------------
# Model training:
tree_model.fit(X_train, y_train)
forrest_model.fit(X_train, y_train)
#---------------------------------------------------------------------
# Model prediction:
tree_predictions = tree_model.predict(X_test)
forrest_predictions = forrest_model.predict(X_test)
#---------------------------------------------------------------------
# Accuracy measurement:
from sklearn.metrics import accuracy_score
tree_score = accuracy_score(y_test, tree_predictions)
print("DecisionTree Accuracy Score:", tree_score)
forrest_score = accuracy_score(y_test, tree_predictions)
print("RandomForrest Accuracy Score:", forrest_score)
#---------------------------------------------------------------------
# Feature importance:
importance = tree_model.feature_importances_
print("Importance of [Experience, Salary, Performance] in DecisionTree:", importance)
importance = forrest_model.feature_importances_
print("Importance of [Experience, Salary, Performance] in RandomForrest:", importance)
#---------------------------------------------------------------------
# Showing decision tree:
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
plot_tree(tree_model, feature_names=["experience","salary","performance"], filled=True)
plt.show()
