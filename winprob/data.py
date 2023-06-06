import fastf1 as ff1
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from datetime import datetime

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

race_dict = {'Abu_Dhabi_Grand_Prix': 22, 'Australian_Grand_Prix': 3,
             'Austrian_Grand_Prix': 11, 'Azerbaijan_Grand_Prix': 8, 'Bahrain_Grand_Prix': 1, 'Belgian_Grand_Prix': 14,
             'Brazilian_Grand_Prix': 21, 'British_Grand_Prix': 10, 'Canadian_Grand_Prix': 9,
             'Dutch_Grand_Prix': 15, 'Emilia_Romagna_Grand_Prix': 4, 'French_Grand_Prix': 12,
             'Hungarian_Grand_Prix': 13, 'Italian_Grand_Prix': 16, 'Japanese_Grand_Prix': 18,
             'Mexico_City_Grand_Prix': 20, 'Miami_Grand_Prix': 5, 'Monaco_Grand_Prix': 7,
             'Saudi_Arabian_Grand_Prix': 2, 'Singapore_Grand_Prix': 17,
             'Spanish_Grand_Prix': 6, 'United_States_Grand_Prix': 19}

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
    return circuits


def get_race(year, track, fn):
    raw_data = pd.read_json(f'http://ergast.com/api/f1/' + str(year) + '/' + str(track) + '/laps/0.json?limit=1000')
    df = pd.json_normalize(raw_data['MRData']['RaceTable']['Races'])

    df_laps = pd.json_normalize(df['Laps'].values[0], record_path='Timings', meta='number')
    df_laps[['season', 'round']] = df.loc[0, ['season', 'round']]
    df_laps = df_laps[['season', 'round', 'number', 'driverId', 'position', 'time']]
    df_laps.rename(columns={'season': 'year', 'number': 'lap', 'driverId': 'driver', 'time': 'laptime'}, inplace=True)
    df_laps['laptime'] = df_laps['laptime'].apply(lambda row: datetime.strptime(row, '%M:%S.%f').microsecond / 1000000 +
                                                              datetime.strptime(row, '%M:%S.%f').second +
                                                              datetime.strptime(row, '%M:%S.%f').minute * 60)
    df_laps = df_laps.astype({'lap': 'int32', 'position': 'int32'})

    """outliers = ['de_vries', 'hulkenberg']
    if outliers[0] in df_laps['drivers'].unique():
        df_laps = df_laps[df_laps['drivers'] != outliers[0]]
    elif outliers[1] in df_laps['drivers'].unique():
        df_laps = df_laps[df_laps['drivers'] != outliers[1]]"""

    df_laps.to_csv('data/race/' + fn, index=False)

    return df_laps


def get_quali(year, track, fn):
    raw_data = pd.read_json(
        f'http://ergast.com/api/f1/' + str(year) + '/' + str(track) + '/qualifying/0.json?limit=1000')
    df = pd.json_normalize(raw_data['MRData']['RaceTable']['Races'][0]['QualifyingResults'])
    df.drop(columns=['Q1', 'Q2', 'Q3', 'Driver.permanentNumber', 'Driver.code', 'Driver.url',
                     'Driver.givenName', 'Driver.familyName', 'Driver.dateOfBirth',
                     'Driver.nationality', 'Constructor.constructorId', 'Constructor.url',
                     'Constructor.name', 'Constructor.nationality'], inplace=True)
    df.rename(columns={'Driver.driverId': 'driver'}, inplace=True)
    df.fillna(0, inplace=True)
    """for q in ['Q1', 'Q2', 'Q3']:
        print(q)
        df[q] = df[q].apply(lambda row: datetime.strptime(row, '%M:%S.%f').microsecond / 1000000 +
                                        datetime.strptime(row, '%M:%S.%f').second +
                                        datetime.strptime(row, '%M:%S.%f').minute * 60)"""
    df = df.astype({'position': 'int32'})

    """outliers = ['de_vries', 'hulkenberg']
    if outliers[0] in df['drivers'].unique():
        df = df[df['drivers'] != outliers[0]]
    elif outliers[1] in df['drivers'].unique():
        df = df[df['drivers'] != outliers[1]]"""


    df.to_csv('data/quali/' + fn, index=False)

    return df


def plot_laps(X):
    lap_nums = [(i + 1) for i in range(len(X))]
    # print(lap_nums)
    plt.plot(lap_nums, X)
    plt.xticks(np.arange(min(lap_nums), max(lap_nums) + 1, 5.0))
    plt.xlabel("Lap Number")
    plt.ylabel("Lap Time (s)")
    plt.title("")
    plt.show()
