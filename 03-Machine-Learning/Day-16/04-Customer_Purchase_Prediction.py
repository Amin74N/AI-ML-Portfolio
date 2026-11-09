# Importing data:
import pandas as pd
data = {"age":[18,25,35,45,55],"income":[1000,2000,4000,6000,8000],"bought":[0,0,1,1,1]}
df = pd.DataFrame(data)
#---------------------------------------------------------------------
# Showing raw data:
print("Dataset:\n", df)
#---------------------------------------------------------------------
# Target/Feature split:
X = df[['age','income']]
y = df.bought
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
# Model prediction + probability:
probpredictions = model.predict_proba(pd.DataFrame([[30,3500]], columns=['age','income']))
print("Probability of buying for a 30-year-old person with $3500 income is:")
print(probpredictions, "[[NotBuy Buy]]")
#---------------------------------------------------------------------
# Accuracy measurement:
predictions = model.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, predictions)
print("Accuracy Score:", accuracy)
#---------------------------------------------------------------------
# ConfusionMatrix:
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, predictions)
print("Confusion Matrix:")
print(cm)
