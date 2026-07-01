# Warning Prevention:
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
# ----------------------------------------------------------------------------------------------------------------------
# Dataset:
import pandas as pd
df = pd.read_csv('students.csv')
# ----------------------------------------------------------------------------------------------------------------------
# X/y Split:
X = df[['hours', 'attendance']]
y = df['passed']
# ----------------------------------------------------------------------------------------------------------------------
# Train/Test Split:
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state= 42)
# ----------------------------------------------------------------------------------------------------------------------
# Model Building:
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
model = Sequential()
# ----------------------------------------------------------------------------------------------------------------------
# Input Layer:
model.add(Input(shape= (2,)))
# ----------------------------------------------------------------------------------------------------------------------
# Hidden Layer:
model.add(Dense(50, activation= "relu"))
# ----------------------------------------------------------------------------------------------------------------------
# Output Layer:
model.add(Dense(1, activation= "sigmoid"))
# ----------------------------------------------------------------------------------------------------------------------
# Model Compiling:
model.compile(optimizer= "adam", loss= "binary_crossentropy", metrics= ["accuracy"])
# ----------------------------------------------------------------------------------------------------------------------
# Model Training:
model.fit(X_train, y_train, epochs= 100, validation_data= (X_test, y_test))
# ----------------------------------------------------------------------------------------------------------------------
# Making predictions:
print("Passing Probability:", model.predict(pd.DataFrame([[7, 90]], columns= ['hours', 'attendance'])))
