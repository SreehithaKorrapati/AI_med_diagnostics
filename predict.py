import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_model(filename='heart_disease_model.pkl'):
    """
    Load the saved model from a file.
    """
    model = joblib.load(filename)
    return model


def preprocess_input_data(input_data):
    """
    Preprocess new input data (same preprocessing as training data).
    """
    # Example preprocessing steps for input_data (must be a pandas DataFrame)
    # Assuming input_data contains 'sex' as a categorical column
    input_data['sex'] = input_data['sex'].map({'Female': 0, 'Male': 1})

    # Feature scaling using StandardScaler
    scaler = StandardScaler()
    input_data_scaled = scaler.fit_transform(input_data)
    return input_data_scaled


def predict_new_data(input_data):
    """
    Predict heart disease for new input data.
    """
    model = load_model()
    input_data_scaled = preprocess_input_data(input_data)
    prediction = model.predict(input_data_scaled)
    return prediction


# Example usage
if __name__ == "__main__":
    # New data to predict (as an example)
    new_data = pd.DataFrame({
        'age': [63], 'sex': ['Male'], 'cp': [3], 'trestbps': [145], 'chol': [233],
        'fbs': [1], 'restecg': [0], 'thalach': [150], 'exang': [0], 'oldpeak': [2.3],
        'slope': [3], 'ca': [0], 'thal': [1]
    })

    prediction = predict_new_data(new_data)
    print("Prediction (0 = No Heart Disease, 1 = Heart Disease):", prediction)
