import pandas as pd
df = pd.read_csv("Car Price Prediction.csv")
pd.set_option('display.max_columns', None)
print(df)
print("---------------------------------------------------------------------")
# Finding null units:
df.drop(columns= ['name'], inplace = True)
print(df.isnull().sum())
print("---------------------------------------------------------------------")
# Encoding categorical data:
from sklearn.preprocessing import OrdinalEncoder
cols = ['fuel','seller_type','transmission','owner']
encoder = OrdinalEncoder()
df[cols] = encoder.fit_transform(df[cols]).astype(int)
print(df)
print("---------------------------------------------------------------------")
# Normalizing numeric data:
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
num_cols = ['year','selling_price','km_driven']
df[num_cols] = scaler.fit_transform(df[num_cols])
print(df)
print("---------------------------------------------------------------------")
# Splitting train/test data:
from sklearn.model_selection import train_test_split
x = df.drop('selling_price', axis = 1)
y = df['selling_price']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)
print("Train/Test Data Split")
print("---------------------------------------------------------------------")
# Linear regression:
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
lg = LinearRegression()
lg.fit(x_train, y_train)
ypred = lg.predict(x_test)
print(r2_score(y_test, ypred))
print("---------------------------------------------------------------------")
# Random forest regression:
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
model = RandomForestRegressor(random_state= 42)
model.fit(x_train, y_train)
ypred = model.predict(x_test)
print(r2_score(y_test, ypred))
