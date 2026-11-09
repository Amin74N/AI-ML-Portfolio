# Importing data:
import pandas as pd
data = {"age":[20,22,25,28,30,33,35,37,40,43,45,48,50,55],
        "income":[2750,3000,3250,3500,3750,4000,4250,4500,4750,5000,5250,5500,5750,6000],
        "buy":[0,0,1,0,0,1,0,1,1,0,1,1,1,1]}
df = pd.DataFrame(data)
# ---------------------------------------------------------------------
# Target/Feature split:
X = df.drop(columns= ['buy'])
y = df['buy']
# ---------------------------------------------------------------------
# Train/Test split:
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
# ---------------------------------------------------------------------
# Model building:
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100)
# ---------------------------------------------------------------------
# Model training:
model.fit(X_train, y_train)
# ---------------------------------------------------------------------
# Train/Test Accuracy Score:
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)
print("Train Accuracy Score:", train_score)
print("Test Accuracy Score:", test_score)
# ---------------------------------------------------------------------
# Classification Criteria Scores:
from sklearn.metrics import (precision_score,recall_score,f1_score)
pred = model.predict(X_test)
print("Precision Score:", precision_score(y_test,pred))
print("Recall Score:", recall_score(y_test,pred))
print("F1 Score:", f1_score(y_test,pred))
# ---------------------------------------------------------------------
# Cross Validation Score:
from sklearn.model_selection import cross_val_score
cvscores = cross_val_score(model,X,y,cv=5)
print("Cross Validation Score:", cvscores.mean())
#---------------------------------------------------------------------
# ConfusionMatrix:
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, pred)
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure()
sns.heatmap(cm, annot=True)
plt.xlabel("Predicted")
plt.ylabel("Truth")
plt.show()
