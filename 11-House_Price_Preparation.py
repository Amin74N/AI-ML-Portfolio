# Importing data:
import pandas as pd
df = pd.read_csv("House Price.csv")
# ---------------------------------------------------------------------
# Showing raw data:
print("Raw Data:\n", df)
# ---------------------------------------------------------------------
# Target/Feature split:
x = df.drop(columns=['price'])
y = df['price']
#---------------------------------------------------------------------
# Encoding categorical data:
df = pd.get_dummies(x).astype(int)
#---------------------------------------------------------------------
# Scaling numerical data (Normalization):
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
num_cols = ['area','rooms']
df[num_cols] = scaler.fit_transform(df[num_cols])
#---------------------------------------------------------------------
# Showing the result:
print("\nResult:\n", pd.concat([df, y], axis=1))
