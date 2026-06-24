# Importing data:
import pandas as pd
df = pd.read_csv('customer_churn_dataset.csv')
#---------------------------------------------------------------------
# Showing raw data:
print("Dataset Columns:\n", dir(df))
#---------------------------------------------------------------------
# Managing empty units:
df = df.dropna(subset=['Churn'])
df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Usage Frequency'] = df['Usage Frequency'].fillna(df['Usage Frequency'].mean())
#---------------------------------------------------------------------
# Target/Feature split:
X = df[['Age','Usage Frequency']]
y = df.Churn
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
# ---------------------------------------------------------------------
# Accuracy measurement:
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, predictions)
print("Accuracy Score:", accuracy)
#---------------------------------------------------------------------
# ConfusionMatrix:
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
