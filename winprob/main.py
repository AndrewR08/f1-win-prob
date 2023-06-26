from data import *
from train import *
from predict import *
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

    # add user input
    year = 2023
    skip_race = 'Miami_Grand_Prix'

    skip_files = [str(year) + '_' + skip_race + '_R.csv', str(year) + '_' + skip_race + '_Q.csv']

    # not_raced = get_all_races(year, race_dict)

    races_dir = "data/" + str(year) + "/race/"
    quali_dir = "data/" + str(year) + "/quali/"
    r_out_fn = "data/" + str(year) + "_races.csv"
    q_out_fn = "data/" + str(year) + "_quali.csv"

    # combine_csv(races_dir, r_out_fn)
    # combine_csv(quali_dir, q_out_fn)

    X_final, y_final, yw_final = create_mult_dataset(races_dir, quali_dir, skip_files)

    train(year, skip_race, X_final, y_final, yw_final)

    predicted = predict(year, skip_race)

    plot_positions(year, skip_race, drivers=[])

    plot_single_prob(year, skip_race, predicted, prob_lap=8)
    plot_probs(year, skip_race, predicted)


if __name__ == '__main__':
    main()
