# Importing data:
from sklearn.datasets import load_digits
digits = load_digits()
# ---------------------------------------------------------------------
# Target/Feature split:
X = digits['data']
y = digits['target']
print("Target Classes:", digits.target_names)
# ---------------------------------------------------------------------
# Train/Test split:
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
# ---------------------------------------------------------------------
# Model building:
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=1000)
# ---------------------------------------------------------------------
# Model training:
model.fit(X_train, y_train)
# ---------------------------------------------------------------------
# Calculating Train/Test Accuracy Score:
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)
# ---------------------------------------------------------------------
# Showing Train/Test Accuracy Scores:
print("Train Accuracy Score:", train_score)
print("Test Accuracy Score:", test_score)
# ---------------------------------------------------------------------
# Loading Classification Criteria:
from sklearn.metrics import (precision_score,recall_score,f1_score)
pred = model.predict(X_test)
# ---------------------------------------------------------------------
# Showing Classification Criteria Scores:
print("Precision Score:", precision_score(y_test,pred, average='weighted'))
print("Recall Score:", recall_score(y_test,pred, average='weighted'))
print("F1 Score:", f1_score(y_test,pred, average='weighted'))
