import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score

import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras import backend as K


def get_data(path):
    return pd.read_csv(path, sep=';')


def age_to_years(dataset):
    dataset['age'] = dataset['age'] / 365
    dataset['age'] = dataset['age'].astype(np.int64)


def gender_mod(dataset):
    dataset.loc[dataset['gender'] == 1, 'gender'] = 'female'
    dataset.loc[dataset['gender'] == 2, 'gender'] = 'male'


def data_select(dataset):
    X = dataset.iloc[:, 1:-1].values
    y = dataset.iloc[:, -1].values

    return X, y


def gender_encod(X):
    return LabelEncoder().fit_transform(X[:, 1])


def std_scaler():
    return StandardScaler()


def scaler_fit(X_train, X_test):
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    return X_train, X_test


def model_splitting(X, y, test):
    return train_test_split(X, y, test_size=test, random_state=42)


def ann_creation():
    ann = tf.keras.models.Sequential()
    return ann


def first_layer(ann, qty, act):
    ann.add(tf.keras.layers.Dense(units=qty, activation=act))


def add_layer(ann, qty, act):
    ann.add(tf.keras.layers.Dense(units=qty, activation=act))


def batch_norm(ann, ax, mmt):
    ann.add(tf.keras.layers.BatchNormalization(axis=ax, momentum=mmt))


def layers_dropout(rate):
    tf.keras.layers.Dropout(rate)


def output_layer(ann, act):
    ann.add(tf.keras.layers.Dense(units=1, activation=act))


def ann_compile(ann, opt, lss, mts):
    ann.compile(optimizer=opt, loss=lss, metrics=mts)


def ann_training(ann, X_train, y_train, btch, epch):
    ann.fit(X_train, y_train, batch_size=btch, epochs=epch, workers=-1, use_multiprocessing=True,
            validation_data=(X_test, y_test))


def recall(y_test, y_pred):
    true_positives = K.sum(K.round(K.clip(y_test * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_test, 0, 1)))
    recall_keras = true_positives / (possible_positives + K.epsilon())
    return recall_keras


def precision(y_test, y_pred):
    true_positives = K.sum(K.round(K.clip(y_test * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision_keras = true_positives / (predicted_positives + K.epsilon())
    return precision_keras


def f1(y_test, y_pred):
    p = precision(y_test, y_pred)
    r = recall(y_test, y_pred)
    return 2 * ((p * r) / (p + r + K.epsilon()))


if __name__ == '__main__':
    dataset = get_data(path='cardio_train.csv')

    age_to_years(dataset)

    gender_mod(dataset)

    X, y = data_select(dataset)

    X[:, 1] = gender_encod(X)

    X_train, X_test, y_train, y_test = model_splitting(X, y, test=0.30)

    sc = std_scaler()

    X_train, X_test = scaler_fit(X_train, X_test)

    ann = ann_creation()

    first_layer(ann, qty=16, act='relu')

    # batch_norm(ann, ax=-1, mmt=0.99)

    add_layer(ann, qty=8, act='relu')

    # layers_dropout(0.5)

    # add_layer(ann, qty=64, act='relu')
    #
    # add_layer(ann, qty=32, act='relu')

    output_layer(ann, act='relu')

    y_pred = ann.predict(X_test)
    ann_compile(ann, opt='adam', lss='binary_crossentropy', mts=['accuracy', precision,
                                                                 recall, f1])

    ann_training(ann, X_train, y_train, btch=32, epch=250)
