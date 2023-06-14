import keras.backend
import keras
from keras.models import Sequential
from keras.layers import *
from keras.callbacks import EarlyStopping, ModelCheckpoint
from data import *
from sklearn.model_selection import train_test_split
from keras.models import load_model


# function to load model from training if it exists, otherwise print ERROR message
# - file_path: path to model.h5 file
def load(file_path):
    if os.path.exists(file_path):
        best_model = load_model(file_path)
    else:
        print("ERROR: File Not Found")
    return best_model


def main():
    cache(True)

    year = 2022
    race_dict = get_schedule(year)
    print(race_dict)

    #placeholder for race to predict on (therefore not included in training)
    skip_race = 'Spanish_Grand_Prix'
    skip_files = [str(year) + '_' + skip_race + '_R.csv', str(year) + '_' + skip_race + '_Q.csv']

    #not_raced = get_all_races(year, race_dict)

    races_dir = "data/" + str(year) + "/race/"
    quali_dir = "data/" + str(year) + "/quali/"
    r_out_fn = "data/" + str(year) + "_races.csv"
    q_out_fn = "data/" + str(year) + "_quali.csv"

    #combine_csv(races_dir, r_out_fn)
    #combine_csv(quali_dir, q_out_fn)

    X_final, y_final, yw_final = create_mult_dataset(races_dir, quali_dir, skip_files)

    X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.2, random_state=8)

    #X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25)

    print(X_final.shape)
    print(y_final.shape)
    print(X_final)

    train = True
    model_name = str(year) + "races" + "_no_" + skip_race
    if train:
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
        model.add(Dense(units=64, activation='linear'))
        model.add(Dense(units=16, activation='linear'))
        model.add(Dense(units=2, activation='linear'))
        model.add(Dense(units=X_train.shape[1]-1, activation='softmax'))

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

        #evaluate model with validation data and when always predicting leader to win
        print()
        best_model.evaluate(X_test, y_test)
        best_model.evaluate(X_final, yw_final)

    predict = True
    if predict:
        print("---- PREDICT ----")
        p_year = 2022
        p_track = 'Spanish_Grand_Prix'

        p_df = pd.read_csv('data/'+str(p_year)+'/race/'+str(p_year)+'_'+p_track+'_R.csv')
        p_qdf = pd.read_csv('data/'+str(p_year)+'/quali/'+str(p_year)+'_'+p_track+'_Q.csv')

        Xp, yp, yp_win = create_dataset(p_df, p_qdf)

        # need to load model from file
        pred_model = load_model('best_models/2022races_no_'+p_track+'.h5')

        pred_model.evaluate(Xp, yp)
        pred_model.evaluate(Xp, yp_win)

        predicted = pred_model.predict(Xp)
        pred = np.argmax(predicted, axis=1)
        print(pred[:20])


if __name__ == '__main__':
    main()
