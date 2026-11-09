# Importing data:
import pandas as pd
data = {"age":[20,22,25,28,30,33,35,37,40,43,45,48,50,55],
        "income":[2750,3000,3250,3500,3750,4000,4250,4500,4750,5000,5250,5500,5750,6000],
        "buy":[0,0,1,0,0,1,0,1,1,0,1,1,1,1]}
df = pd.DataFrame(data)
#---------------------------------------------------------------------
# Showing raw data:
print(df)
#---------------------------------------------------------------------
# Target/Feature split:
X = df.drop(columns= ['buy'])
y = df['buy']
#---------------------------------------------------------------------
# Train/Test split:
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
#---------------------------------------------------------------------
# Model building:
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100)
#---------------------------------------------------------------------
# Model training:
model.fit(X_train, y_train)
#---------------------------------------------------------------------
# Probability prediction:
sampleprediction = model.predict_proba(pd.DataFrame([[27,6000]], columns=['age', 'income']))
print("Does a 27-year-old person with $6000 income buy? [[No Yes]] =", sampleprediction)
#---------------------------------------------------------------------
# Model prediction:
predictions = model.predict(X_test)
#---------------------------------------------------------------------
# Accuracy measurement:
from sklearn.metrics import accuracy_score
score = accuracy_score(y_test, predictions)
print("Accuracy Score:", score)
#---------------------------------------------------------------------
# Feature importance:
importance = model.feature_importances_
print("Importance of Age & Income:", importance)
