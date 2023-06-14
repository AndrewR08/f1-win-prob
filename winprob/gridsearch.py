import itertools
import keras
from keras.models import *
from keras.layers import *
from keras.losses import *
from random import shuffle
from tensorflow import keras
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, EarlyStopping
from data import *
from sklearn.model_selection import train_test_split


# fix random seed for reproducibility
np.random.seed(8)
tf.random.set_seed(8)

# Set initial parameters
optimizer = 'adam'
batch_size = 8
epochs = 50

# Debug settings
PRINT_PERMUTATIONS = True  # Whether to print the amount of permutations while running
RANDOM_ORDERING = True  # Whether to grid search in random order (good for faster discovery)


def run(layers, loss_function, optimizer, batch_size, epochs, patience=15):
    # Clear backend
    keras.backend.clear_session()

    # Debug variables
    layers_str = "[" + "|".join(str(str(x.units) + " " + x._keras_api_names[0][13:]) for x in layers) + "]"
    loss_function_name = loss_function.name
    print(f"hyper-parameters:\n\t" +
          f"Layers: {layers_str}\n\tLoss Function: {loss_function_name}\n\tBatch Size: {batch_size}\n\t" +
          f"Epochs: {epochs}")

    # Setup path for artifacts
    output_path = 'best_models/gridsearch.h5'

    races_dir = "data/2022/race/"
    quali_dir = "data/2022/quali/"

    X_final, y_final, yw_final = create_mult_dataset(races_dir, quali_dir)
    X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.2, random_state=8)

    # Setup callbacks
    model_checkpoint = ModelCheckpoint(output_path, monitor='val_loss',  mode='min', verbose=1, save_best_only=True)
    early_stopping = EarlyStopping(monitor='val_loss', patience=patience, verbose=1)

    # Sequential Model
    model = Sequential()
    model.add(Input(shape=(X_train.shape[1]), name='input'))

    # Hidden Layers
    for layer in layers:
        model.add(layer)

    # Add output layer
    model.add(Dense(units=X_train.shape[1] - 1, activation='softmax'))

    # Compile model
    model.optimizer = optimizer
    model.compile(loss=loss_function, metrics=['accuracy'])
    model.summary()

    # Fit model to training data
    history = model.fit(X_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_data=[X_test, y_test],
                        callbacks=[model_checkpoint])
    number_of_epochs_ran = len(history.history['val_loss'])
    val_loss = model.evaluate(X_test, y_test, verbose=0)
    print("----val loss-----", val_loss)

    # Write result to results csv
    csv_result = f"{layers_str},{loss_function_name},{batch_size},{epochs},{number_of_epochs_ran},{val_loss}\n"
    file1 = open('results/results_gs.csv', 'a+')
    file1.write(csv_result)
    file1.close()
    print("Results appended.\n")


def grid_search(layer_types, layer_counts, neuron_counts, loss_functions):
    if PRINT_PERMUTATIONS:
        amt_loss_functions = len(loss_functions)
        amt_layer_types = len(layer_types)
        amt_neuron_counts = len(neuron_counts)
        amt_total = 0
        for layer_count in layer_counts:
            amt_neuron_total = amt_neuron_counts ** layer_count
            amt_activation_total = amt_layer_types ** layer_count
            amt_total += (amt_neuron_total * amt_activation_total)
        amt_total *= amt_loss_functions
        print(f"Total permutations: {amt_total}")

    layer_permutations = []
    print("Calcuating permutations...")
    for loss_function in loss_functions:
        for layer_count in layer_counts:
            neuron_count_permutations = list(itertools.product(neuron_counts, repeat=layer_count))
            neuron_activation_permutations = list(itertools.product(layer_types, repeat=layer_count))
            perms = list(itertools.product(neuron_count_permutations, neuron_activation_permutations))
            for layer_neuron_counts, activations in perms:
                layers = []
                for i in range(len(layer_neuron_counts)):
                    neuron_amt = layer_neuron_counts[i]
                    activation = activations[i]
                    layer_name = "layer" + str(len(layers))
                    layers.append(Dense(neuron_amt, activation=activation, name=layer_name))
                layer_permutations.append(layers)
    amt_layer_permutations = len(layer_permutations)
    print(f"All {amt_layer_permutations} permutations compiled.")

    if RANDOM_ORDERING:
        print("Randomizing permutation order...")
        shuffle(layer_permutations)
        print("Randomized.")
    print("Beginning grid search...")
    for layer_permutation in layer_permutations:
        run(layer_permutation, loss_function, optimizer, batch_size, epochs)
    print("Grid search complete.")


def main():
    # Set hyperparameters for Grid Search
    layer_types = ['relu', 'linear']
    layer_counts = [1, 2, 3]
    neuron_counts = [2, 4, 8, 16, 32, 64, 128, 256]
    loss_functions = [SparseCategoricalCrossentropy()]

    # Run gridsearch function
    grid_search(layer_types, layer_counts, neuron_counts, loss_functions)


if __name__ == '__main__':
    main()
