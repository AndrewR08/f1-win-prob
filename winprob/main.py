import cmd
from train import *
from predict import *


def main():
    # location of cache for fastf1 - True for pc, False for mac
    cache(True)

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
        inv_race_dict = {v: k for k, v in race_dict.items()}

        print()

        if year == 2023:
            current_round = 7
            # using list comprehension to perform in one line
            race_dict = {v: k for k, v in inv_race_dict.items() if not (isinstance(v, int) and (v > current_round))}

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
        skip_race = 'Australian_Grand_Prix'

    print()
    print(year, skip_race)
    print()

    skip_files = [str(year) + '_' + skip_race + '_R.csv', str(year) + '_' + skip_race + '_Q.csv']

    races_dir = "data/" + str(year) + "/race/"
    quali_dir = "data/" + str(year) + "/quali/"

    X_final, y_final, yw_final = create_mult_dataset(races_dir, quali_dir, skip_files)

    model_name = str(year) + "races" + "_no_" + skip_race
    model_path = 'best_models/' + model_name + '.h5'

    # Check if model for race already exists, if not train new model
    if not load(model_path):
        train(model_name, X_final, y_final, yw_final)

    # Call predict function to use trained model to make win prob predictions
    predicted = predict(year, skip_race)

    # plotting functions
    #plot_positions(year, skip_race, drivers=[])
    #plot_single_prob(year, skip_race, predicted, prob_lap=8)
    #plot_probs(year, skip_race, predicted)

    plot_pos_and_probs(year, skip_race, X_final, predicted)

    """
    -- TO DO --
    For each test race, please provide a visual showing the actual position of each driver and their probability of 
    winning for each lap. Maybe the height of a line indicates the driver's position, and the width of the line could 
    represent the probability. -> could create subplot w/ existing positions plot (filtered to pred laps) & plot probs

    Across all laps of all test races, please aggregate the predicted probabilities to see how well they match win 
    probabilities. For example, for how many driver-laps did the model output 95-100% win probability, and what 
    percentage of the time did the driver actually win? Then repeat for 90-94.9, 85-89.9, etc. --> need to save 
    predicted probs and aggregate
    """


if __name__ == '__main__':
    main()
