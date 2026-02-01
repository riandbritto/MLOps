# Import necessary libraries
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

if __name__ == '__main__':
    # Load the Iris dataset
    diabetes = load_diabetes()
    X, y = diabetes.data, diabetes.target

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a Random Forest classifier
    model = LinearRegression(random_state=42)
    model.fit(X_train, y_train)

    # Save the model to a file
    joblib.dump(model, 'diabetes_model.pkl')
    
    print("The model training was successful")
