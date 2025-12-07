import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class MeraLR:
    def __init__(self):
        self.m = None
        self.b = None

    def fit(self, X_train, y_train):
        # ensure numpy arrays of floats
        X = np.array(X_train, dtype=float)
        y = np.array(y_train, dtype=float)

        # compute means once (avoid recomputing inside loop)
        x_mean = X.mean()
        y_mean = y.mean()

        num = 0.0
        den = 0.0
        for i in range(X.shape[0]):
            num += (X[i] - x_mean) * (y[i] - y_mean)
            den += (X[i] - x_mean) ** 2    # <-- denominator is sum of (x - x_mean)^2

        if den == 0:
            raise ValueError("Denominator is zero — all X values are equal.")

        self.m = num / den
        self.b = y_mean - self.m * x_mean

    def predict(self, X_test):
        # works for scalar or array-like input
        X = np.array(X_test, dtype=float)
        return self.m * X + self.b

# -----------------------------
# Using the class with your CSV
df = pd.read_csv("C:\\Users\\Dell\\OneDrive\\Desktop\\NARUTO\\datasets\\Placement.csv")
X = df.iloc[:, 0].values
y = df.iloc[:, 1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

print("X_train shape:", X_train.shape)   # NOTE: .shape not callable

lr = MeraLR()
lr.fit(X_train, y_train)

print("m (slope):", lr.m)
print("b (intercept):", lr.b)

# Predict a single value (scalar) or whole array
single_pred = lr.predict(X_test[0])          # scalar prediction
batch_pred = lr.predict(X_test)             # numpy array of predictions

print("single input:", X_test[0], "-> pred:", single_pred)
print("y_pred shape:", batch_pred.shape) 
