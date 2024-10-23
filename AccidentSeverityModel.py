# Required Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Load dataset
df = pd.read_csv(r'C:\Users\DERRICK_CALVINCE\Desktop\PR\python projects\accidentseveritymodel\your_dataset.csv')

# Drop rows with missing values
df = df.dropna()

# Convert categorical features to numerical using LabelEncoder
le = LabelEncoder()
for col in df.select_dtypes(include=['object']).columns:
    df[col] = le.fit_transform(df[col])

# Define the features (independent variables) and target (dependent variable)
X = df.drop(['Accident_severity'], axis=1)  # Features
y = df['Accident_severity']  # Target

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# Save the model to a file for future use
joblib.dump(model, 'accident_severity_model.pkl')

# Example of using the model to predict accident severity for new data
# Ensure the new example data has the same number of features as the training set
example_data = np.array([[15, 2, 1, 1, 2, 3, 2, 1, 1, 4, 0, 3, 1, 1, 0, 1, 2, 1, 1, 2, 1, 1, 0, 0, 1, 0, 1, 2, 1, 1, 0]]).reshape(1, -1)

# Predict the accident severity for the new example data
predicted_severity = model.predict(example_data)
print(f"Predicted Accident Severity: {predicted_severity[0]}")
