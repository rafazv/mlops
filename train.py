import os
import random
import numpy as np
import random as python_random
import tensorflow
import tensorflow as tf
import dagshub
import mlflow
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, InputLayer
from keras.utils import to_categorical

import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def reset_seeds() -> None:
  os.environ['PYTHONHASHSEED']=str(42)
  tf.random.set_seed(42)
  np.random.seed(42)
  random.seed(42)

def read_data():
    url = 'raw.githubusercontent.com'
    username = 'renansantosmendes'
    repository = 'lectures-cdas-2023'
    file_name = 'fetal_health_reduced.csv'
    data = pd.read_csv(f'https://{url}/{username}/{repository}/master/{file_name}')
    X=data.drop(["fetal_health"], axis=1)
    y=data["fetal_health"]

    return X, y

def process_data(X, y):
    columns_names = list(X.columns)
    scaler = preprocessing.StandardScaler()
    X_df = scaler.fit_transform(X)
    X_df = pd.DataFrame(X_df, columns=columns_names)

    X_train, X_test, y_train, y_test = train_test_split(X_df,
                                                        y,
                                                        test_size=0.3,
                                                        random_state=42)

    y_train = y_train -1
    y_test = y_test - 1

    return X_train, X_test, y_train, y_test

def create_model(X):
    reset_seeds()
    model = Sequential()
    model.add(InputLayer(shape=(X.shape[1], )))
    model.add(Dense(units=10, activation='relu'))
    model.add(Dense(units=10, activation='relu'))
    model.add(Dense(units=3, activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

    return model

def config_mlflow():
    dagshub.init(repo_owner='rafazv', repo_name='my-first-repo', mlflow=True)
    mlflow.tensorflow.autolog(log_models=True,
                              log_input_examples=True,
                              log_model_signatures=True)

def train_model(model, X, y, is_train=True):
    with mlflow.start_run():
      model.fit(X,
                y,
                epochs=50,
                validation_split=0.2,
                verbose=3)

if __name__ == '__main__':
    X, y = read_data()
    X_train, X_test, y_train, y_test = process_data(X, y)
    model = create_model(X_train)
    config_mlflow()
    train_model(model, X_train, y_train)