from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

import data_loader


def train_mlp(X_train, y_train):
    mlp = MLPClassifier(
        hidden_layer_sizes=(64, 64), max_iter=360, random_state=1, activation="relu"
    )

    mlp.fit(X_train, y_train)
    return mlp


if __name__ == '__main__':
    data = data_loader.load_data(include_rep=False)
    df_train, df_test = data_loader.split_data(data, sample=[(30, 360), (21, 40)], test_bound=40)

    X_train = df_train.drop(columns=['subject']).to_numpy()
    y_train = df_train['subject'].to_numpy()
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    mlp = train_mlp(X_train, y_train)
    X_test = df_test.drop(columns=['subject']).to_numpy()
    y_test = df_test['subject'].to_numpy()
    X_test = scaler.transform(X_test)

    accuracy = mlp.score(X_test, y_test)
    print(f"Accuracy: {accuracy * 100:.2f}%")
