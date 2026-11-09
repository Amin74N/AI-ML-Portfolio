# Importing data:
import pandas as pd
data = {"area":[50,70,90,120,150,170,200,220,250,300],
        "price":[100,150,200,300,400,450,520,600,700,850]}
df = pd.DataFrame(data)
# ---------------------------------------------------------------------
# Showing raw data:
print("Dataset:\n", df)
#---------------------------------------------------------------------
# Showing EDA:
print("Brief EDA:\n", df.describe())
#---------------------------------------------------------------------
# Target/Feature split:
X = df.drop(columns= ['price'])
y = df['price']
#---------------------------------------------------------------------
# Train/Test split:
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
#---------------------------------------------------------------------
# Model building:
from sklearn.linear_model import LinearRegression
model = LinearRegression()
#---------------------------------------------------------------------
# Model training:
model.fit(X_train, y_train)
#---------------------------------------------------------------------
# Model prediction:
prediction = model.predict(X_test)
#---------------------------------------------------------------------
# Mean Absolute Error measurement:
from sklearn.metrics import mean_absolute_error
predictions = model.predict(X_test)
error = mean_absolute_error(y_test, predictions)
print("MAE:",error)
#---------------------------------------------------------------------
# R2 Score measurement:
from sklearn.metrics import r2_score
score = r2_score(y_test, predictions)
print("R^2:",score)
