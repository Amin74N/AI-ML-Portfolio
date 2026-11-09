# Data Loading:
from Preprocessing import X_train, y_train
# ----------------------------------------------------------------------------------------------------------------------
# Model Building:
from sklearn.linear_model import LogisticRegression
LR_model = LogisticRegression(solver= "liblinear", max_iter= 5000)
from sklearn.ensemble import RandomForestClassifier
RF_model = RandomForestClassifier(n_estimators= 100, random_state= 42)
from sklearn.tree import DecisionTreeClassifier
DT_model = DecisionTreeClassifier(random_state= 42)
# ----------------------------------------------------------------------------------------------------------------------
# Model Training:
LR_model.fit(X_train,y_train)
RF_model.fit(X_train,y_train)
DT_model.fit(X_train,y_train)
