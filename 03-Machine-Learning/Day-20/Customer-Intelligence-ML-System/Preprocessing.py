# Data Loading:
import pandas as pd
df = pd.read_csv("customer_clean.csv")
# ----------------------------------------------------------------------------------------------------------------------
# Feature Engineering:
df = pd.get_dummies(df)
X = df.drop(["index", "purchased"], axis= 1)
y = df["purchased"]
# ----------------------------------------------------------------------------------------------------------------------
# Train/Test Split:
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size= 0.2, random_state= 42)
# ----------------------------------------------------------------------------------------------------------------------
# Data Scaling:
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
