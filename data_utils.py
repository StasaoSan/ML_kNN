import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split


def load_and_preprocess_data(csv_path, target_column, test_size=0.2, val_size=0.25, random_state=42):
    df = pd.read_csv(csv_path)

    X = df.drop(columns=[target_column]).values
    y = df[target_column].values

    le = LabelEncoder()
    y = le.fit_transform(y)
    y = y.astype(float)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=val_size,
                                                      random_state=random_state)

    return X_train, X_val, X_test, y_train, y_val, y_test, le, scaler
