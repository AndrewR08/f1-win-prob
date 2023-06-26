from data import *
from train import load
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def predict(year, skip_race):
    # load race and qualifying csv to create dataset
    p_df = pd.read_csv('data/' + str(year) + '/race/' + str(year) + '_' + skip_race + '_R.csv')
    p_qdf = pd.read_csv('data/' + str(year) + '/quali/' + str(year) + '_' + skip_race + '_Q.csv')

    # use data.py method to create dataset
    Xp, yp, yp_win = create_dataset(p_df, p_qdf)

    # load model from file
    pred_model = load('best_models/' + str(year) + 'races_no_' + skip_race + '.h5')

    # number of laps to predict with
    # - pred_laps[0] = start lap
    # - pred_laps[1] = end lap
    pred_laps = [0, len(Xp)]

    # get user input
    pred_laps[0] = int(input("Enter prediction start lap (1-" + str(len(Xp)) + "): "))
    pred_laps[1] = int(input("Enter prediction end lap (1-" + str(len(Xp)) + "): "))

    if pred_laps[0] < 0:
        print("ERROR - start input out of range, using 1")
        pred_laps[0] = 1
    elif pred_laps[1] > len(Xp):
        print("ERROR - end input out of range, using ", len(Xp))
        pred_laps[1] = len(Xp)
    else:
        # ability to print out trained/base model evaluations
        """
        print("\nTrained Model: ")
        pred_model.evaluate([Xp[pred_laps[0]-1:pred_laps[1]]], [yp[pred_laps[0]-1:pred_laps[1]]])
        print("Base Model: ")
        pred_model.evaluate([Xp[pred_laps[0]-1:pred_laps[1]]], [yp_win[pred_laps[0]-1:pred_laps[1]]])
        """

        # call model.predict to return win probabilities for the number of input laps
        predicted = pred_model.predict(Xp[pred_laps[0]-1:pred_laps[1]])

        # array to show which driver class was predicted to win at each lap
        pred = np.argmax(predicted, axis=1)

        return predicted
