from main import load
from data import *


def predict(year, skip_race):
    print("---- PREDICT ----")
    p_year = year
    p_track = skip_race

    p_df = pd.read_csv('data/' + str(p_year) + '/race/' + str(p_year) + '_' + p_track + '_R.csv')
    p_qdf = pd.read_csv('data/' + str(p_year) + '/quali/' + str(p_year) + '_' + p_track + '_Q.csv')

    Xp, yp, yp_win = create_dataset(p_df, p_qdf)
    print(Xp[0])
    print(yp[0])
    print(yp_win[0])

    # load model from file
    pred_model = load('best_models/' + str(p_year) + 'races_no_' + p_track + '.h5')

    # number of laps to predict with -- 1 = quali data --> len() = full race
    pred_laps = 15
    # -- could add specific lap to predict on (ex. lap 25/50 instead of full first 25 laps)
    if pred_laps > len(Xp):
        print("ERROR - input laps must be less than ", len(Xp))
    else:
        pred_model.evaluate([Xp[:pred_laps]], [yp[:pred_laps]])
        pred_model.evaluate([Xp[:pred_laps]], [yp_win[:pred_laps]])

        predicted = pred_model.predict(Xp[:pred_laps])
        pred = np.argmax(predicted, axis=1)
        print(pred)

        return predicted