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
        ys.append(x2.index(1))

    X = np.array(Xs)  # , dtype=np.float64)
    y = np.array(ys)  # , dtype=np.float64)

    X = Xs
    y = ys
    """for row in X:
        print(row)
    print(y)"""

    return X, y


def get_all_races(year, tracks):
    for t in tracks:
        print(t)
        rf = str(year) + '_' + str(t) + '_R.csv'
        qf = str(year) + '_' + str(t) + '_Q.csv'
        t_num = race_dict[t]
        df1 = get_race(year, t_num, rf)
        df2 = get_quali(year, t_num, qf)


def combine_csv(csvs_dir, out_dir):

    csv_files = os.listdir(csvs_dir)
    print(csv_files)

    combined_df = pd.DataFrame()

    for file in csv_files:
        print(file)
        df = pd.read_csv(csvs_dir+file)
        combined_df = combined_df.append(df, ignore_index=True)

    combined_df.to_csv(out_dir, index=False)


def main():
    year = 2022
    tracks = list(race_dict.keys())

    #get_all_races(year, tracks)

    races_dir = "data/race/"
    quali_dir = "data/quali/"
    r_out_fn = "data/" + str(year) + "_races.csv"
    q_out_fn = "data/" + str(year) + "_quali.csv"

    #combine_csv(races_dir, r_out_fn)
    #combine_csv(quali_dir, q_out_fn)

    df = pd.read_csv(r_out_fn)
    print(df)
    q_df = pd.read_csv(q_out_fn)

    uniq_drivers = sorted(q_df['driver'].unique())
    print(uniq_drivers)

    r_files = os.listdir(races_dir)
    q_files = os.listdir(quali_dir)

    X_all = []
    y_all = []
    for rf in r_files:
        df = pd.read_csv(races_dir + rf)
        for qf in q_files:
            q_df = pd.read_csv(quali_dir + qf)

        X, y = create_dataset(df, q_df)
        X_all.append(X)
        y_all.append(y)

    X_new = [item for sublist in X_all for item in sublist]
    X_final = np.array(X_new)
    y_new = [item for sublist in y_all for item in sublist]
    y_final = np.array(y_new)

    X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.2, random_state=8)

    print(X_final.shape)
    print(y_final.shape)
    print(X_final)

    train = True
    if train:
        keras.backend.clear_session()

        model = Sequential()
        model.add(Input(shape=(None, X_train.shape[1])))
        model.add(Dense(units=128, activation='relu'))
        model.add(Dense(units=X_train.shape[1]-1, activation='softmax'))

        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        model.fit(X_train, y_train, epochs=100)

        model.evaluate(X_test, y_test)


if __name__ == '__main__':
    main()
