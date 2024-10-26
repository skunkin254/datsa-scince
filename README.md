1. Dataset Selection and Preparation

    Define Variables:
        Dependent variable (target): This is usually a measure of accident severity (e.g., "Severity" or "Injury Level").
        Independent variables (predictors): Factors that influence accident severity, such as "Weather Condition," "Road Type," "Vehicle Speed," "Lighting Condition," etc.

2. Building the Linear Regression Model

Using Python libraries like Pandas for data manipulation and scikit-learn for machine learning,  set up the model:  
python                                                                                         
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib

# Load the dataset
df = pd.read_csv('path_to_your_dataset.csv')

# Define dependent and independent variables
X = df[['Weather Condition', 'Road Type', 'Vehicle Speed', 'Lighting Condition']]  # Example columns
y = df['Severity']  # Dependent variable

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Test the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Save the model for future use
joblib.dump(model, 'accident_severity_model.pkl')
3. Predicting Accident Severity (Example)
use it to predict the severity of an accident based on hypothetical values: 
# Load the model
model = joblib.load('accident_severity_model.pkl')

# Example input for prediction
new_data = pd.DataFrame({
    'Weather Condition': [1],  # Hypothetical input value
    'Road Type': [2],
    'Vehicle Speed': [60],
    'Lighting Condition': [3]
})

predicted_severity = model.predict(new_data)
print("Predicted Accident Severity:", predicted_severity[0])
4. Benefits of the Model
•	Accident Analysis: Helps in identifying high-risk factors and improving road safety by analyzing accident patterns.
•	Resource Allocation: Authorities in underdeveloped countries can focus on high-risk areas and conditions.
•	Preventive Measures: Insights gained can be used to make informed decisions on road improvements, traffic regulations, and driver awareness programs. 


