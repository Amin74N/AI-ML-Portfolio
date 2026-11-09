# Importing data:
from sklearn.datasets import load_iris
iris = load_iris()
#---------------------------------------------------------------------
# Showing raw data:
print("Dataset Columns:\n", dir(iris))
#---------------------------------------------------------------------
# Target/Feature split:
X = iris.data
y = iris.target
print("Target Classes:", iris.target_names)
#---------------------------------------------------------------------
# Train/Test split:
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)
#---------------------------------------------------------------------
# Model building:
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
#---------------------------------------------------------------------
# Model training:
model.fit(X_train, y_train)
#---------------------------------------------------------------------
# Model prediction:
predictions = model.predict(X_test)
#---------------------------------------------------------------------
# Accuracy measurement:
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, predictions)
print("Accuracy Score:", accuracy)
#---------------------------------------------------------------------
# ConfusionMatrix:
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, predictions)
print("Confusion Matrix:")
print(cm)
