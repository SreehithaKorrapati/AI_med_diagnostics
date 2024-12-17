from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def split_data(df):
    # Split features and target variable
    X = df.drop('target', axis=1)
    y = df['target']

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


def scale_features(X_train, X_test):
    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled
