# Data Loading:
import joblib
model = joblib.load("Best_RF_Model.pkl")
# ----------------------------------------------------------------------------------------------------------------------
# Prediction: [[Age, Income, Score, Rasht City, Shiraz City, Tabriz City, Tehran City]]
new_customer = [[35,5000,60,1,0,0,0]]
print("New Customer Will Buy" if model.predict(new_customer)[0] == 1 else "New Customer Will Not Buy")