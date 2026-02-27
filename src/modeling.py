import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def train_and_save_model(df, model_path="models/random_forest.pkl"):

    # Encode target
    le = LabelEncoder()
    df['target'] = le.fit_transform(df['next_day_bucket'])

    X = df[['Size USD', 'Leverage', 'trade_count']]
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    joblib.dump(model, model_path)

    return model