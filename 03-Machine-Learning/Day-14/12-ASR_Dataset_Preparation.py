# Importing data:
import pandas as pd
df = pd.read_csv("ASR_dataset.csv")
pd.set_option('display.max_columns', None)
# ---------------------------------------------------------------------
# Showing raw data:
print("Raw Data:\n", df)
#---------------------------------------------------------------------
# Encoding categorical data:
df = pd.get_dummies(df).astype(int)
#---------------------------------------------------------------------
# Scaling numerical data (Normalization):
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
num_cols = ['duration']
df[num_cols] = scaler.fit_transform(df[num_cols])
#---------------------------------------------------------------------
# Showing the result:
print("\nResult:\n", df)
