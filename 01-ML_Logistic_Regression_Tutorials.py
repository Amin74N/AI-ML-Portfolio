# Importing data:
from sklearn.datasets import load_digits
digits = load_digits()
#---------------------------------------------------------------------
# Showing raw data:
print("Dataset Columns:\n", dir(digits))
#---------------------------------------------------------------------
# Target/Feature split:
X = digits['data']
y = digits['target']
print("Target Classes:", digits.target_names)
#---------------------------------------------------------------------
# Train/Test split:
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
#---------------------------------------------------------------------
# Model building:
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=1000)
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
# ConfusionMatrix building:
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, predictions)
#---------------------------------------------------------------------
# ConfusionMatrix Visualization:
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure()
sns.heatmap(cm, annot=True)
plt.xlabel("Predicted")
plt.ylabel("Truth")
plt.show()
