data = {"age":[25,30,35,40],
        "gender":["Male","Female","Male","Female"],
        "income":[3000,4000,5000,6000]}
#---------------------------------------------------------------------
# Importing data:
import pandas as pd
df = pd.DataFrame(data)
# ---------------------------------------------------------------------
# Showing raw data:
print("Raw Data:\n", df)
#---------------------------------------------------------------------
# Encoding categorical data:
df = pd.get_dummies(df).astype(int)
#---------------------------------------------------------------------
# Scaling numerical data (Standardization):
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
num_cols = ['age','income']
df[num_cols] = scaler.fit_transform(df[num_cols])
#---------------------------------------------------------------------
# Showing the result:
print("\nResult:\n", df)
