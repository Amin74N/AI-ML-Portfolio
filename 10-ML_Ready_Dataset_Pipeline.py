# Importing data:
import pandas as pd
df = pd.read_csv("customer_dataset.csv")
# ---------------------------------------------------------------------
# Showing raw data:
print("Raw Data:\n", df)
# ---------------------------------------------------------------------
# Target/Feature split:
x = df.drop(columns=['name','purchased'])
y = df['purchased'].map({'Yes':1, 'No':0})
#---------------------------------------------------------------------
# Encoding categorical data:
df = pd.get_dummies(x).astype(int)
#---------------------------------------------------------------------
# Scaling numerical data (Standardization):
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
num_cols = ['age','income']
df[num_cols] = scaler.fit_transform(df[num_cols])
#---------------------------------------------------------------------
# Showing the result:
print("\nResult:\n", pd.concat([df, y], axis=1))
