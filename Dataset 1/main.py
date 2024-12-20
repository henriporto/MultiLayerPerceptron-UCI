import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from ucimlrepo import fetch_ucirepo
import joblib

RANDOM_STATE = 24


def load():
    car_evaluation = fetch_ucirepo(id=19)

    # pandas dataframes
    X = car_evaluation.data.features
    y = car_evaluation.data.targets
    return X, y


def preprocess(X, y):
    # categorias por feature
    categories = [
        ['vhigh', 'high', 'med', 'low'],   # buying
        ['vhigh', 'high', 'med', 'low'],   # maint
        ['2', '3', '4', '5more'],          # doors
        ['2', '4', 'more'],                # persons
        ['small', 'med', 'big'],           # lug_boot
        ['low', 'med', 'high']             # safety
    ]

    # one-hot encoding
    encoder = OneHotEncoder(categories=categories, sparse_output=False, dtype=int)
    X_encoded = encoder.fit_transform(X)

    # convertendo as classes categoricas em inteiros
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y.values.ravel())

    return X_encoded, y_encoded, label_encoder


def split(X, y):
    # dividindo em 70% treino, 30% teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
    )
    return X_train, X_test, y_train, y_test


def train(X_train, y_train):
    # usei os valores de hiperparametros padroes do scikit-learn: https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html#mlpclassifier
    model = MLPClassifier(random_state=RANDOM_STATE, max_iter=500)
    model.fit(X_train, y_train)
    if model.n_iter_ == model.max_iter:
        print("opa, o modelo nao convergiu (aumentar 'max_iter')")
    return model


def evaluate(model, X_test, y_test, label_encoder):
    y_pred = model.predict(X_test)

    # voltando as classes originais
    y_pred_decoded = label_encoder.inverse_transform(y_pred)
    y_test_decoded = label_encoder.inverse_transform(y_test)

    # calculando metricas
    accuracy = accuracy_score(y_test_decoded, y_pred_decoded)
    precision = precision_score(y_test_decoded, y_pred_decoded, average='weighted', zero_division=0)
    recall = recall_score(y_test_decoded, y_pred_decoded, average='weighted', zero_division=0)
    f1 = f1_score(y_test_decoded, y_pred_decoded, average='weighted', zero_division=0)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("Classification Report:")
    print(classification_report(y_test_decoded, y_pred_decoded, zero_division=0))


def save_model(model):
    joblib.dump(model, 'model.pkl')


if __name__ == '__main__':
    X, y = load()
    X_encoded, y_encoded, label_encoder = preprocess(X, y)
    X_train, X_test, y_train, y_test = split(X_encoded, y_encoded)
    model = train(X_train, y_train)
    evaluate(model, X_test, y_test, label_encoder)
    save_model(model)
