# Importing data:
import pandas as pd
data = {"hours":[1,2,3,4,5,6,7,8], "passed":[0,0,0,1,1,1,1,1]}
df = pd.DataFrame(data)
#---------------------------------------------------------------------
# Showing raw data:
print("Dataset:\n", df)
#---------------------------------------------------------------------
# Target/Feature split:
X = df[['hours']]
y = df['passed']
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
probpredictions = model.predict_proba(pd.DataFrame([[6]], columns=['hours']))
print("Probability of passing with 6 hours is:")
print(probpredictions, "[[NotPass Pass]]")
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
