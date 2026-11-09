# Importing data:
import pandas as pd
data = pd.read_csv('headbrain.csv')
# ---------------------------------------------------------------------
# Showing raw data:
print(data)
#---------------------------------------------------------------------
# Target/Feature split:
X = data.drop(columns= ['Brain Weight(grams)'])
y = data['Brain Weight(grams)']
#---------------------------------------------------------------------
# Train/Test split:
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
#---------------------------------------------------------------------
# Model building:
from sklearn.linear_model import LinearRegression
model = LinearRegression()
#---------------------------------------------------------------------
# Model training:
model.fit(X_train, y_train)
#---------------------------------------------------------------------
# Model prediction:
predictions = model.predict(X_test)
#---------------------------------------------------------------------
# Accuracy measurement:
from sklearn.metrics import r2_score
r2_score = r2_score(y_test, predictions)
#---------------------------------------------------------------------
# Showing the Result:
print(r2_score)
