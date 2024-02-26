import numpy as np
import matplotlib.pyplot as plt


def mouse_run_on_track():
    return

def population_code(neuron_spiking):
    esimates = np.sum(neuron_spiking, axis=0)
    return esimates

def main():

    learning_rate = 0.01 # percentage of neural states to change per iteration
    n_neurons = 100
    sims = 10000

    track_length = 100 # vu
    sampling_rate = 100 # Hz
    speed = 10 # vu / s

    samples_across_track = int(sampling_rate*(track_length/speed))
    true_positions = np.linspace(0, track_length, samples_across_track)

    neuron_spiking = np.random.choice([0, 1], size=(n_neurons, len(true_positions)))
    error = 1e111
    for i in range(sims):
        print("Error in population code:", str(error))
        pred_positions = population_code(neuron_spiking)
        error = np.sum(np.abs(pred_positions-true_positions))

        # random perturbation of the network to make a mutant network
        perturbation_mask = np.random.choice([0, 1], size=(n_neurons, len(true_positions)), p=[1-learning_rate, learning_rate])
        tmp = np.random.choice([0, 1], size=(n_neurons, len(true_positions)))

        mutant_neuron_spiking = neuron_spiking.copy()
        mutant_neuron_spiking[perturbation_mask] = tmp[perturbation_mask]
        mutant_pred_positions = population_code(mutant_neuron_spiking)
        mutant_error = np.sum(np.abs(mutant_pred_positions - true_positions))

        if mutant_error < error:
            neuron_spiking = mutant_neuron_spiking.copy() # evolve
            error = mutant_error

    print("done")

if __name__ == '__main__':
    main()