# Importing data:
import pandas as pd
data = {"experience":[1,2,3,4,5,6],
        "salary":[2000,2500,3000,3500,4000,4500]}
df = pd.DataFrame(data)
# ---------------------------------------------------------------------
# Showing raw data:
print("Dataset:\n", df)
#---------------------------------------------------------------------
# Target/Feature split:
X = df.drop(columns= ['salary'])
y = df['salary']
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
prediction = model.predict(pd.DataFrame([[8]], columns=X.columns))
#---------------------------------------------------------------------
# Showing the Result:
print("Salary for 8 years experience:", prediction)
