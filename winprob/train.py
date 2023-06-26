import keras.backend
import keras
from keras.models import Sequential
from keras.layers import *
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from main import load


def train(year, skip_race, X_final, y_final, yw_final):
    X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.2, random_state=8)

    print(X_final.shape)
    print(y_final.shape)
    print(X_final)
    print(X_train.shape)

    to_train = True
    model_name = str(year) + "races" + "_no_" + skip_race
    if to_train:
        # define patience used for early stopping and initialize early stopping / best model saving
        patience = 15
        early_stopping = EarlyStopping(monitor='val_loss', patience=patience, verbose=1)

        model_checkpoint = ModelCheckpoint('best_models/' + model_name + '.h5', monitor='val_loss', mode='min',
                                           verbose=1,
                                           save_best_only=True)

        # clear previous model training data to ensure best model outcomes
        keras.backend.clear_session()

        # define sequential model with linear Dense layer and softmax Dense output layer
        model = Sequential()
        model.add(Input(shape=(None, X_train.shape[1])))
        #model.add(Dense(units=16, activation='linear'))
        #model.add(Dense(units=16, activation='linear'))
        model.add(Dense(units=X_train.shape[1] - 1, activation='softmax'))

        # include how to calculate accuracy in slides
        # keras.metrics.SparseTopKCategoricalAccuracy(k=3) <-- used for right answer within top 3 probs
        # compile model using sparse cat crossentropy as loss function and adam as optimizer
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        # fit model using training data, can use high epochs w/ early stopping
        # - epochs: 50 (early stopping will override this)
        # - batch_size: 8 (best fit from testing)
        # - validation_dat: X_test, y_test (created w/ train test split)
        # - callbacks: early stopping and model checkpoint saving
        model.fit(X_train, y_train,
                  epochs=50,
                  batch_size=8,
                  validation_data=[X_test, y_test],
                  callbacks=[model_checkpoint, early_stopping])

        model_path = 'best_models/' + model_name + '.h5'
        best_model = load(model_path)

        # evaluate model with validation data and when always predicting leader to win
        print()
        best_model.evaluate(X_test, y_test)
        best_model.evaluate(X_final, yw_final)