from scripts.data_preprocessing import load_data, handle_missing_values, encode_categorical
from scripts.feature_engineering import split_data, scale_features
from scripts.model import train_model, tune_hyperparameters, evaluate_model, save_model

def main():
    # Load and preprocess the data
    df = load_data('C:/Users/kavit/PycharmProjects/heart_disease_prediction/data/heart.csv')
    df = handle_missing_values(df)
    df = encode_categorical(df)

    # Split and scale the data
    X_train, X_test, y_train, y_test = split_data(df)
    X_train_scaled, X_test_scaled = scale_features(X_train, X_test)

    # Hyperparameter tuning and model training
    model = tune_hyperparameters(X_train_scaled, y_train)

    # Evaluate the model
    evaluate_model(model, X_test_scaled, y_test)

    # Save the tuned model
    save_model(model)

if __name__ == "__main__":
    main()
