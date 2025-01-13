import cmd
from pprint import pprint

from torch_train import *
from train import *
from predict import *

def main():
    # location of cache for fastf1 - True for pc, False for mac
    cache(pc=True)
    print()

    valid_years = [2022, 2023, 2024]
    schedule_dict = {}
    race_dict = {}
    current_date = datetime(2023, 7, 10)  # Replace with datetime.now()

    # download files for all valid races
    for year in valid_years:
        schedule = get_schedule(year)
        schedule_dict[year] = schedule
        valid_races = get_valid_races(schedule, current_date)
        race_dict[year] = {i + 1: race for i, race in enumerate(valid_races.values())}
        get_all_race_data(year, race_dict)

    pprint(race_dict)

    # used for development, whether to display ui or use hard-coded values
    ui = False
    if ui:
        # Display year options menu
        year_options = {2022: 1, 2023: 2}
        year_keys = list(year_options.keys())
        for i in range(len(year_options)):
            year_keys[i] = str(i+1) + " - " + str(year_keys[i])
        
        cli = cmd.Cmd()
        cli.columnize(year_keys, displaywidth=80)

        # Get user input for year using year dictionary
        inv_year_dict = {v: k for k, v in year_options.items()}
        year_input = int(input("Select Year (1-" + str(len(year_keys)) + "): "))
        year = inv_year_dict[year_input]
        
        race_dict = get_schedule(year)
        
        # Specify the date (use datetime.now() for the actual current date)
        current_date = datetime(2024, 7, 10)  # Replace with datetime.now() or a variable
        past_races = get_valid_races(race_dict, current_date)

        print("Races up to the current date:")
        for k, v in past_races.items():
            print(f"{k}: {v['name']} (Date: {v['date']})")

        # Display race options menu
        r_keys = list(race_dict.values())
        for i in range(len(r_keys)):
            r_keys[i] = str(i + 1) + " - " + r_keys[i]
        cli = cmd.Cmd()
        cli.columnize(r_keys, displaywidth=80)

        # Get user input for track using track dictionary
        race_input = int(input("Select Race (1-" + str(len(r_keys)) + "): "))
        skip_race = race_dict[race_input]
    else:
        year = 2023
        skip_race = 'Spanish_Grand_Prix'

    print("\n", year, skip_race, "\n")

    skip_files = [str(year) + '_' + skip_race + '_R.csv', str(year) + '_' + skip_race + '_Q.csv']

    races_dir = "data/" + str(year) + "/race/"
    quali_dir = "data/" + str(year) + "/quali/"

    X_final, y_final, yw_final = create_mult_dataset(races_dir, quali_dir, skip_files)

    # model_name = str(year) + "races" + "_no_" + skip_race
    # model_path = '../best_models/' + model_name + '.h5'

    model = NeuralNetwork(input_size=X_final.shape[1])
    model_name = str(year) + "races" + "_no_" + skip_race
    model_path = 'best_models/' + model_name + '.pth'  # Use .pth extension for PyTorch

    # Check if model for race already exists, if not train new model
    if not load_torch(model_path, model):
        torch_train(model_name, X_final, y_final, yw_final)

    # Call predict function to use trained model to make win prob results
    predicted = predict(year, skip_race, ui)

    # plotting functions
    plot_positions(year, skip_race, drivers=[])
    plot_single_prob(year, skip_race, predicted, prob_lap=8)
    plot_probs(year, skip_race, predicted)
    plot_pos_and_probs(year, skip_race, predicted)


if __name__ == '__main__':
    main()
