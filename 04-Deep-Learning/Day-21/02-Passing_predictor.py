# Warning Prevention:
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
# ----------------------------------------------------------------------------------------------------------------------
# Dataset:
import numpy as np
X = np.array([[1],[2],[3],[4],[5],[6]])
y = np.array([0,0,0,1,1,1])
# ----------------------------------------------------------------------------------------------------------------------
# Model Building:
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
model = Sequential()
# ----------------------------------------------------------------------------------------------------------------------
# Input Layer:
model.add(Input(shape=(1,)))
# ----------------------------------------------------------------------------------------------------------------------
# Hidden Layer:
model.add(Dense(20, activation= "relu"))
# ----------------------------------------------------------------------------------------------------------------------
# Output Layer:
model.add(Dense(1, activation= "sigmoid"))
# ----------------------------------------------------------------------------------------------------------------------
# Model Compiling:
model.compile(optimizer= "adam", loss= "binary_crossentropy", metrics= ["accuracy"])
# ----------------------------------------------------------------------------------------------------------------------
# Model Training:
model.fit(X, y, epochs= 100)
# ----------------------------------------------------------------------------------------------------------------------
# Making predictions:
print("Passing Probability:", model.predict(np.array([[7]])))
