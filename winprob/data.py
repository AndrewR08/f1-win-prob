import fastf1 as ff1
from fastf1 import plotting
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from datetime import datetime
import os

warnings.simplefilter(action='ignore', category=FutureWarning)
pd.options.mode.chained_assignment = None  # default='warn'

lap_dict = {'Abu_Dhabi_Grand_Prix': 5554, 'Australian_Grand_Prix': 5303,
            'Austrian_Grand_Prix': 4318, 'Azerbaijan': 6003, 'Bahrain_Grand_Prix': 5412, 'Belgian_Grand_Prix': 7004,
            'Brazilian_Grand_Prix': 4309, 'British_Grand_Prix': 5891, 'Canadian_Grand_Prix': 4361,
            'Dutch_Grand_Prix': 4259, 'Emilia_Romagna_Grand_Prix': 4909, 'French_Grand_Prix': 5842,
            'Hungarian_Grand_Prix': 4381, 'Italian_Grand_Prix': 5793, 'Japanese_Grand_Prix': 5807,
            'Mexico_City_Grand_Prix': 4304, 'Miami_Grand_Prix': 5410, 'Monaco_Grand_Prix': 3337,
            'Saudi_Arabian_Grand_Prix': 6175, 'Singapore_Grand_Prix': 5063,
            'Spanish_Grand_Prix': 4655, 'United_States_Grand_Prix': 5513}

drivers_dict = {1: 'VER', 3: 'RIC', 4: 'NOR', 5: 'VET', 6: 'LAT', 10: 'GAS', 11: 'PER', 14: 'ALO', 16: 'LEC', 18: 'STR',
                20: 'MAG', 22: 'TSU', 23: 'ALB', 24: 'ZHO', 31: 'OCO', 44: 'HAM', 47: 'MSC', 55: 'SAI', 63: 'RUS',
                77: 'BOT'}


# function to determine cache location of fastf1 data
# - pc: True = Desktop, False = Mac
def cache(pc):
    if pc:
        # location of cache for pc
        ff1.Cache.enable_cache('D:/f1data')
    else:
        # location of cache for mac
        ff1.Cache.enable_cache('/Users/andrewreeves/Documents/ASU/fastf1')


def format_circuits(circuit):
    if isinstance(circuit, str):
        circuit = circuit.replace(' ', '_')
    return circuit


def get_schedule(year):
    schedule = ff1.get_event_schedule(year)
    schedule = schedule[schedule['EventFormat'] != 'testing']
    circuits = schedule.EventName
    circuits = circuits.apply(format_circuits).reset_index(drop=True)
    circuits = circuits.to_dict()
    circuits = {k + 1: v for k, v in circuits.items()}
    return circuits


def get_race(year, track, fn, skip_list):
    raw_data = pd.read_json(f'http://ergast.com/api/f1/' + str(year) + '/' + str(track) + '/laps/0.json?limit=1000')

    try:
        df = pd.json_normalize(raw_data['MRData']['RaceTable']['Races'])
        df_laps = pd.json_normalize(df['Laps'].values[0], record_path='Timings', meta='number')
        df_laps[['season', 'round']] = df.loc[0, ['season', 'round']]
        df_laps = df_laps[['season', 'round', 'number', 'driverId', 'position', 'time']]
        df_laps.rename(columns={'season': 'year', 'number': 'lap', 'driverId': 'driver', 'time': 'laptime'},
                       inplace=True)
        df_laps['laptime'] = df_laps['laptime'].apply(
            lambda row: datetime.strptime(row, '%M:%S.%f').microsecond / 1000000 +
                        datetime.strptime(row, '%M:%S.%f').second +
                        datetime.strptime(row, '%M:%S.%f').minute * 60)
        df_laps = df_laps.astype({'lap': 'int32', 'position': 'int32'})

        df_laps.to_csv('data/' + str(year) + '/race/' + fn, index=False)
    except KeyError:
        skip_list.append(track)

    return skip_list


def get_quali(year, track, fn):
    raw_data = pd.read_json(
        f'http://ergast.com/api/f1/' + str(year) + '/' + str(track) + '/qualifying/0.json?limit=1000')
    try:
        df = pd.json_normalize(raw_data['MRData']['RaceTable']['Races'][0]['QualifyingResults'])
        df.drop(columns=['Q1', 'Q2', 'Q3', 'Driver.permanentNumber', 'Driver.code', 'Driver.url',
                         'Driver.givenName', 'Driver.familyName', 'Driver.dateOfBirth',
                         'Driver.nationality', 'Constructor.constructorId', 'Constructor.url',
                         'Constructor.name', 'Constructor.nationality'], inplace=True)
        df.rename(columns={'Driver.driverId': 'driver'}, inplace=True)
        df.fillna(0, inplace=True)
        df = df.astype({'position': 'int32'})

        """outliers = ['de_vries', 'hulkenberg']
        if outliers[0] in df['drivers'].unique():
            df = df[df['drivers'] != outliers[0]]
        elif outliers[1] in df['drivers'].unique():
            df = df[df['drivers'] != outliers[1]]"""

        df.to_csv('data/' + str(year) + '/quali/' + fn, index=False)

    except IndexError:
        pass


def create_dataset(df, q_df):
    drivers = q_df['driver'].unique()
    sorted_drivers = sorted(drivers)
    # print("in create dataset: ", sorted_drivers)

    laps = df['lap'].max()

    Xs = []
    ys = []
    y_win = []
    # laps = 1
    for i in range(0, laps + 1):
        x1 = [None] * (len(sorted_drivers) + 1)
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
                # print(d_ind)
                x1[d_ind] = pos[0]
            else:
                try:
                    pos = df['position'].loc[(df['lap'] == i) & (df['driver'] == d)].values
                    d_ind = sorted_drivers.index(d) + 1
                    # print(d_ind)
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
        ys.append(Xs[-1].index(1) - 1)

    X = Xs
    y = ys

    return X, y, y_win


def get_all_races(year, race_dict):
    not_raced = []
    for t_num, t in race_dict.items():
        rf = str(year) + '_' + str(t) + '_R.csv'
        qf = str(year) + '_' + str(t) + '_Q.csv'
        not_raced = get_race(year, t_num, not_raced, rf)
        get_quali(year, t_num, qf)
    return not_raced


def combine_csv(csvs_dir, out_dir):
    csv_files = os.listdir(csvs_dir)
    combined_df = pd.DataFrame()

    for file in csv_files:
        df = pd.read_csv(csvs_dir + file)
        combined_df = combined_df.append(df, ignore_index=True)

    combined_df.to_csv(out_dir, index=False)


def create_mult_dataset(races_dir, quali_dir, skip_files=None):
    if skip_files is None:
        skip_files = ['', '']
    else:
        pass

    r_files = os.listdir(races_dir)
    q_files = os.listdir(quali_dir)

    X_all = []
    y_all = []
    yw_all = []
    for rf in r_files:
        if rf == skip_files[0]:
            pass
        else:
            df = pd.read_csv(races_dir + rf)
            for qf in q_files:
                if qf == skip_files[1]:
                    pass
                else:
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

    return X_final, y_final, yw_final


def plot_positions(year, track_name, drivers=None):
    ff1.plotting.setup_mpl()

    r_file = 'data/' + str(year) + '/race/' + str(year) + "_" + track_name + "_R.csv"
    q_file = 'data/' + str(year) + '/quali/' + str(year) + "_" + track_name + "_Q.csv"
    df = pd.read_csv(r_file)
    qdf = pd.read_csv(q_file)

    # create a matplotlib figure
    fig = plt.figure(figsize=[8, 6], dpi=200)
    ax = fig.add_subplot()

    if drivers is None or drivers == []:
        drivers = list(qdf.driver.unique())

    min_y = []
    max_y = []
    for d in drivers:
        x = [0]
        y = qdf['position'].loc[qdf['driver'] == d].tolist()
        x = x + df['lap'].loc[df['driver'] == d].tolist()
        y = y + df['position'].loc[df['driver'] == d].tolist()
        if len(x) == 0 and len(y) == 0:
            # need to fix drivers who crash on lap 1 not showing up -- single point
            # can compare quali vs race drivers list to get drivers who crash on lap 1 and then skip those in initial
            # plot, append to skip list and then go through skip list after others have been plotted (maybe?)
            ax.scatter(x, y, marker='s', color=plotting.driver_color(d), label=d)
        else:
            ax.plot(x, y, color=plotting.driver_color(d), label=d)
        min_y.append(min(y))
        max_y.append(max(y))

    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.xticks(np.arange(min(x), max(x) + 5, 5.0))
    plt.yticks(np.arange(min(min_y), max(max_y) + 1, 1.0))
    plt.gca().invert_yaxis()
    plt.title(str(year) + " " + track_name.replace("_", " "))
    plt.xlabel("Lap")
    plt.ylabel("Position")
    plt.savefig('images/' + str(year) + "_" + track_name + "_Positions.png")


def plot_single_prob(year, track_name, predictions, prob_lap):
    ff1.plotting.setup_mpl()

    q_file = 'data/' + str(year) + '/quali/' + str(year) + "_" + track_name + "_Q.csv"
    qdf = pd.read_csv(q_file)

    # create a matplotlib figure
    fig = plt.figure(figsize=[8, 6], dpi=200)
    ax = fig.add_subplot()

    drivers = sorted(list(qdf.driver.unique()))

    if prob_lap == -1:
        prob_lap = len(predictions) - 1
    prob_lap = prob_lap - 1
    probs = predictions[prob_lap]
    for i in range(len(drivers)):
        d = drivers[i]
        if year == 2022:
            if d == 'alonso':
                d = 'gasly'
            elif d == 'gasly':
                d = 'de_vries'
            elif d == 'latifi':
                d = 'sargeant'
            elif d == 'mick_schumacher':
                d = 'hulkenberg'
            elif d == 'ricciardo':
                d = 'piastri'
            elif d == 'vettel':
                d = 'alonso'

        p = probs[i]
        ax.bar(i, p, color=plotting.driver_color(d))

    plt.title(str(year) + " " + track_name.replace("_", " ") + " Lap " + str(prob_lap))
    plt.xticks(range(len(drivers)), drivers, rotation='vertical')
    plt.xlabel("Driver")
    plt.ylabel("Win Probability")
    # plt.ylim(0, 0.5)
    plt.tight_layout()
    plt.savefig('images/' + str(year) + "_" + track_name + "_Prediction_L" + str(prob_lap) + ".png")


def plot_probs(year, track_name, predictions):
    ff1.plotting.setup_mpl()

    q_file = 'data/' + str(year) + '/quali/' + str(year) + "_" + track_name + "_Q.csv"
    qdf = pd.read_csv(q_file)

    # create a matplotlib figure
    fig = plt.figure(figsize=[10, 8], dpi=150)
    ax = fig.add_subplot()

    drivers = sorted(list(qdf.driver.unique()))

    predictions = np.transpose(predictions)
    for i in range(len(predictions)):
        d = drivers[i]
        if year == 2022:
            if d == 'alonso':
                d = 'gasly'
            elif d == 'gasly':
                d = 'de_vries'
            elif d == 'latifi':
                d = 'sargeant'
            elif d == 'mick_schumacher':
                d = 'hulkenberg'
            elif d == 'ricciardo':
                d = 'piastri'
            elif d == 'vettel':
                d = 'alonso'

        ax.plot(np.arange(len(predictions[i])), predictions[i], "-o", color=plotting.driver_color(d), label=d)

    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.title(str(year) + " " + track_name.replace("_", " "))
    plt.xlabel("Lap Number")
    plt.ylabel("Win Probability")
    plt.savefig('images/' + str(year) + "_" + track_name + "_Predictions.png")
    #plt.show()


def plot_pos_and_probs(year, track_name, predictions):
    ff1.plotting.setup_mpl()

    q_file = 'data/' + str(year) + '/quali/' + str(year) + "_" + track_name + "_Q.csv"
    r_file = 'data/' + str(year) + '/race/' + str(year) + "_" + track_name + "_R.csv"
    qdf = pd.read_csv(q_file)
    df = pd.read_csv(r_file)

    # Create dataset and remove the first column
    X, y, y_win = create_dataset(df, qdf)
    X_final = np.array(X)
    positions = np.transpose(X_final[:len(predictions), 1:])

    # create a matplotlib figure
    fig, ax = plt.subplots(1, 2, figsize=[20, 6], dpi=150)

    drivers = sorted(list(qdf.driver.unique()))

    predictions = np.transpose(predictions)

    min_y, max_y = [], []
    for i in range(len(predictions)):
        d = drivers[i]
        if year == 2022:
            if d == 'alonso':
                d = 'gasly'
            elif d == 'gasly':
                d = 'de_vries'
            elif d == 'latifi':
                d = 'sargeant'
            elif d == 'mick_schumacher':
                d = 'hulkenberg'
            elif d == 'ricciardo':
                d = 'piastri'
            elif d == 'vettel':
                d = 'alonso'

        # plot poition by lap
        ax[0].plot(np.arange(len(positions[i])), positions[i], color=plotting.driver_color(d), label=d)
        min_y.append(min(positions[i]))
        max_y.append(max(positions[i]))

        # plot predicted probabilities by lap
        ax[1].plot(np.arange(len(predictions[i])), predictions[i], "-o", color=plotting.driver_color(d), label=d)

    ax[0].set_yticks(np.arange(min(min_y), max(max_y) + 1, 1.0))
    ax[0].invert_yaxis()
    ax[0].set_xlabel("Lap")
    ax[0].set_ylabel("Position")
    ax[0].set_title("Driver Position per Lap")

    ax[1].set_xlabel("Lap")
    ax[1].set_ylabel("Probability ")
    ax[1].set_title("Win Probability per Lap")

    # Put a legend to the right of the current axis
    box = ax[1].get_position()
    ax[1].set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))

    fig.suptitle(str(year) + " " + track_name.replace("_", " "))
    plt.savefig('images/' + str(year) + "_" + track_name + "_Probabilities_Predictions.png")
    #plt.show()
