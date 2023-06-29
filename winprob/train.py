import keras.backend
import keras
from keras.models import Sequential
from keras.layers import *
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from keras.models import load_model
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# function to load model from training if it exists, otherwise print ERROR message
# - file_path: path to model.h5 file
def load(file_path):
    if os.path.exists(file_path):
        best_model = load_model(file_path)
    else:
        return False
    return best_model


def train(model_name, X_final, y_final, yw_final):
    X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.2, random_state=8)

    # define patience used for early stopping and initialize early stopping / best model saving
    patience = 15
    early_stopping = EarlyStopping(monitor='val_loss', patience=patience, verbose=1)

    model_checkpoint = ModelCheckpoint('best_models/' + model_name + '.h5', monitor='val_loss', mode='min',
                                       verbose=0,
                                       save_best_only=True)

    # clear previous model training data to ensure best model outcomes
    keras.backend.clear_session()

    # define sequential model with input layer and softmax Dense output layer
    model = Sequential()
    model.add(Input(shape=(None, X_train.shape[1])))
    model.add(Dense(units=X_train.shape[1] - 1, activation='softmax'))

    # compile model
    # - optimizer = 'adam'
    # - loss = 'sparse_categorical_crossentropy'
    # - metrics = ['accuracy']
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # fit model using training data, can use high epochs w/ early stopping
    # - epochs: 50 (early stopping will override this)
    # - batch_size: 8 (best fit from testing)
    # - validation_dat: X_test, y_test (created w/ train test split)
    # - callbacks: early stopping and model checkpoint saving
    # - verbose: 0: no output, 1: output
    model.fit(X_train, y_train,
              epochs=50,
              batch_size=8,
              validation_data=[X_test, y_test],
              callbacks=[model_checkpoint, early_stopping],
              verbose=0)

    model_path = 'best_models/' + model_name + '.h5'
    best_model = load(model_path)

    # ability to evaluate model with validation data and when always predicting leader to win
    """print("\nTrained Model: ")
    best_model.evaluate(X_test, y_test)
    print("Base Model: ")
    best_model.evaluate(X_final, yw_final)"""