import pandas as pd  
import numpy as np  
from sklearn.model_selection import train_test_split  
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load dataset
df = pd.read_csv("C:\\Users\\Dell\\OneDrive\\Desktop\\NARUTO\\datasets\\placement-dataset.csv")

# Clean column names
df.rename(columns=lambda x: x.strip(), inplace=True)

# Handle missing values - drop rows with NaN
df = df.dropna()

# Encode city column (convert text to numbers)
le = LabelEncoder()
df['city'] = le.fit_transform(df['city'])

# Use cgpa, iq, placement as X and city as y
X = df[['cgpa', 'iq', 'placement']]
y = df['city']

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=2)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Coefficients
m = model.coef_
b = model.intercept_

# Metrics
r2 = r2_score(y_test, y_pred)
n = len(y_test)   # number of samples in test set
p = X_test.shape[1]  # number of predictors

# Adjusted R2
adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

results = {
    "MAE": mean_absolute_error(y_test, y_pred),
    "MSE": mean_squared_error(y_test, y_pred),
    "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
    "R2": r2,
    "Adjusted R2": adj_r2
}

results