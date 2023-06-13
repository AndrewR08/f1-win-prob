import os
import keras.backend
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import *
from keras.callbacks import EarlyStopping, ModelCheckpoint
from data import *
from sklearn.model_selection import train_test_split


def create_dataset(df, q_df):
    drivers = q_df['driver'].unique()
    sorted_drivers = sorted(drivers)
    #print("in create dataset: ", sorted_drivers)

    laps = df['lap'].max()

    Xs = []
    ys = []
    y_win = []
    # laps = 1
    for i in range(0, laps + 1):
        x1 = [None] * (len(sorted_drivers)+1)
        """if i == 0:
            drivers = sorted(q_df['driver'].tolist())
        else:
            drivers = sorted(df['driver'].loc[df['lap'] == (i + 1)].tolist())

        if len(drivers) < len(sorted_drivers):
            missing = sorted(list(set(sorted_drivers) - set(drivers)))"""

        x1[0] = laps - i
        for d in sorted_drivers:
            if i == 0:
                pos = q_df['position'].loc[q_df['driver'] == d].values
                d_ind = sorted_drivers.index(d) + 1
                #print(d_ind)
                x1[d_ind] = pos[0]
            else:
                try:
                    pos = df['position'].loc[(df['lap'] == i) & (df['driver'] == d)].values
                    d_ind = sorted_drivers.index(d) + 1
                    #print(d_ind)
                    x1[d_ind] = pos[0]
                except:
                    pass

        x2 = x1.copy()
        x2.pop(0)

        # will fill None with position number
        indices = [i for i, x in enumerate(x1) if x is None]
        for ind in indices:
            max_pos = np.nanmax(np.array(x2, dtype=np.float64)).astype(int)
            x2[ind - 1] = max_pos + 1
            x1[ind] = max_pos + 1

        Xs.append(x1)
        y_win.append(x2.index(1))

    for i in range(len(Xs)):
        ys.append(Xs[-1].index(1)-1)

    # X = np.array(Xs)  # , dtype=np.float64)
    # y = np.array(ys)  # , dtype=np.float64)

    X = Xs
    y = ys

    return X, y, y_win


def get_all_races(year, race_dict):
    skip_list = []
    for t_num, t in race_dict.items():
        rf = str(year) + '_' + str(t) + '_R.csv'
        qf = str(year) + '_' + str(t) + '_Q.csv'
        skip_list = get_race(year, t_num, skip_list, rf)
        get_quali(year, t_num, qf)
    return skip_list


def combine_csv(csvs_dir, out_dir):
    csv_files = os.listdir(csvs_dir)
    combined_df = pd.DataFrame()

    for file in csv_files:
        df = pd.read_csv(csvs_dir+file)
        combined_df = combined_df.append(df, ignore_index=True)

    combined_df.to_csv(out_dir, index=False)


def main():
    cache(True)
    year = 2023
    race_dict = get_schedule(year)
    print(race_dict)

    #skip_list = get_all_races(year, race_dict)

    races_dir = "data/" + str(year) + "/race/"
    quali_dir = "data/" + str(year) + "/quali/"
    r_out_fn = "data/" + str(year) + "_races.csv"
    q_out_fn = "data/" + str(year) + "_quali.csv"

    #combine_csv(races_dir, r_out_fn)
    #combine_csv(quali_dir, q_out_fn)

    df = pd.read_csv(r_out_fn)
    print(df)
    q_df = pd.read_csv(q_out_fn)

    uniq_drivers = sorted(q_df['driver'].unique())
    print(uniq_drivers)

    """r_files = os.listdir(races_dir)
    q_files = os.listdir(quali_dir)

    X_all = []
    y_all = []
    yw_all = []
    for rf in r_files:
        df = pd.read_csv(races_dir + rf)
        for qf in q_files:
            q_df = pd.read_csv(quali_dir + qf)

        X, y, y_win = create_dataset(df, q_df)
        X_all.append(X)
        y_all.append(y)
        yw_all.append(y_win)

    X_new = [item for sublist in X_all for item in sublist]
    X_final = np.array(X_new)
    y_new = [item for sublist in y_all for item in sublist]
    y_final = np.array(y_new)

    yw_new = [item for sublist in yw_all for item in sublist]
    yw_final = np.array(yw_new)
    print(yw_final)

    X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.2, random_state=8)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25)

    print(X_final.shape)
    print(y_final.shape)
    print(X_final)
    print(y_final)"""

    train = False
    if train:
        keras.backend.clear_session()

        model = Sequential()
        model.add(Input(shape=(None, X_train.shape[1])))
        model.add(Dense(units=64, activation='linear'))
        #model.add(Dropout(0.2))
        model.add(Dense(units=X_train.shape[1]-1, activation='softmax'))

        # include how to calculate accuracy in slides
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                      metrics=['accuracy', keras.metrics.SparseTopKCategoricalAccuracy(k=3)])

        # need to save model to h5 file for loading later
        model.fit(X_train, y_train, epochs=50)

        model.evaluate(X_test, y_test)

        model.evaluate(X_final, yw_final)

    predict = False
    if predict:
        print("---- PREDICT ----")
        p_year = 2023
        p_track = 'Saudi_Arabian_Grand_Prix'
        pt_num = race_dict[p_track]
        p_rf = str(p_year) + "_" + p_track + "_R.csv"
        p_qf = str(p_year) + "_" + p_track + "_Q.csv"

        p_df = get_race(p_year, pt_num, p_rf)
        p_qdf = get_quali(p_year, pt_num, p_qf)

        pred_drivers = sorted(p_qdf['driver'].unique())
        print(pred_drivers)

        Xp, yp, yp_win = create_dataset(p_df, p_qdf)

        print(Xp)
        print(yp)
        print(yp_win)
        print(len(yp_win))

        model.evaluate(Xp, yp)

        model.evaluate(Xp, yp_win)

        print("Validation evaluate: ")
        model.evaluate(X_val, y_val)

        # need to load model from file
        predicted = model.predict(Xp)
        pred = np.argmax(predicted, axis=1)

        print(pred[:10])
        print(yp[:10])
        print(yp_win[:10])

        """lap_n = 2
        print(Xp[lap_n])
        print(predicted[lap_n])
        max_prob = max(predicted[lap_n])
        print(max_prob)
        pred_ind = np.where(predicted[lap_n] == max_prob)[0][0]
        print(pred_ind)
        print(pred_drivers[pred_ind])"""


if __name__ == '__main__':
    main()
