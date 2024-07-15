# plot summary data for a given mouse for the vr task
import pandas as pd
import os
import traceback
import sys
import numpy as np
import matplotlib.pyplot as plt
from P2_PostProcess.VirtualReality.plotting import *
from Helpers.array_utility import pandas_collumn_to_numpy_array

def plot_vr_stop_hists(processed_position_data, save_path, title, track_length):
    fig, axs = plt.subplots(5, 6, figsize=(15, 10), sharex=True, sharey=False)
    fig.suptitle(title, fontsize=16)

    days = np.unique(processed_position_data["session_number"])
    for idx, day in enumerate(days):
        ax = axs.flat[idx]
        day_processed_position_data = processed_position_data[processed_position_data["session_number"] == day]

        y_max=0
        bin_size = 5
        for tt in np.unique(processed_position_data["trial_type"]):
            tt_trials = day_processed_position_data[day_processed_position_data["trial_type"] == tt]

            stops = []
            tt_trial_numbers = []
            for i, tn in enumerate(tt_trials["trial_number"]):
                stops.extend(tt_trials["stop_location_cm"].iloc[i])
                tt_trial_numbers.extend(np.ones(len(tt_trials["stop_location_cm"].iloc[i])) * tn)
            stops, tt_trial_numbers = curate_stops(stops, tt_trial_numbers, track_length)

            hist, bin_edges = np.histogram(stops, bins=int(track_length / bin_size), range=(0, track_length), density=True)
            bin_centres = 0.5 * (bin_edges[1:] + bin_edges[:-1])
            y_max=max([y_max, max(hist)])
            ax.plot(bin_centres, hist, '-', color=get_trial_color(tt), linewidth=2)

        ax.set_xlim(0, track_length)
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.set_ylim(0, np.round(y_max+0.01, decimals=2))
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        plot_utility.style_track_plot(ax, track_length)
        plot_utility.style_vr_plot(ax, np.round(y_max+0.01, decimals=2))
        ax.set_yticks([0, np.round(y_max+0.01, decimals=2)])
        ax.set_yticklabels(["", str(np.round(y_max+0.01, decimals=2))])
        ax.set_title(f'Day {day}')

    # Set shared x and y labels
    fig.text(0.5, 0.04, 'Location (cm)', ha='center', fontsize=18)
    fig.text(0.04, 0.5, 'Stops Density', va='center', rotation='vertical', fontsize=18)
    # Adjust layout to prevent overlapping
    plt.tight_layout(rect=[0.03, 0.03, 1, 0.95])
    plt.subplots_adjust(left=0.1, bottom=0.1)
    plt.savefig(save_path + title + '_stop_histograms.png', dpi=200)
    return


def plot_vr_stop_rasters(processed_position_data, save_path, title, track_length):
    fig, axs = plt.subplots(5, 6, figsize=(15, 10), sharex=True, sharey=False)
    fig.suptitle(title, fontsize=16)

    days = np.unique(processed_position_data["session_number"])
    for idx, day in enumerate(days):
        ax = axs.flat[idx]
        day_processed_position_data = processed_position_data[processed_position_data["session_number"] == day]
        for index, trial_row in day_processed_position_data.iterrows():
            trial_row = trial_row.to_frame().T.reset_index(drop=True)
            trial_type = trial_row["trial_type"].iloc[0]
            trial_number = trial_row["trial_number"].iloc[0]
            trial_stop_color = get_trial_color(trial_type)

            ax.plot(np.array(trial_row["stop_location_cm"].iloc[0]),
                    trial_number * np.ones(len(trial_row["stop_location_cm"].iloc[0])),
                    '|', color=trial_stop_color, markersize=4, alpha=0.1)

        ax.set_xlim(0, track_length)
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.set_ylim(0, max(day_processed_position_data["trial_number"]))
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        plot_utility.style_track_plot(ax, track_length)
        plot_utility.style_vr_plot(ax, len(day_processed_position_data)+0.5)
        ax.set_yticks([0, max(day_processed_position_data["trial_number"])])
        ax.set_yticklabels(["", str(max(day_processed_position_data["trial_number"]))])
        ax.set_title(f'Day {day}')

    # Set shared x and y labels
    fig.text(0.5, 0.04, 'Location (cm)', ha='center', fontsize=18)
    fig.text(0.04, 0.5, 'Trial number', va='center', rotation='vertical', fontsize=18)
    # Adjust layout to prevent overlapping
    plt.tight_layout(rect=[0.03, 0.03, 1, 0.95])
    plt.subplots_adjust(left=0.1, bottom=0.1)
    plt.savefig(save_path + title + '_stop_rasters.png', dpi=200)
    return

def process_recordings(vr_recording_path_list):
    all_behaviour = pd.DataFrame()
    for recording in vr_recording_path_list:
        print("processing ", recording)
        try:
            position_data = pd.read_csv(recording+"/processed/position_data.csv")
            processed_position_data = pd.read_pickle(recording+"/processed/processed_position_data.pkl")

            session_id = recording.split("/")[-1]
            mouse_id = session_id.split("_")[0]
            session_number = int(session_id.split("_")[1].split("D")[-1])
            processed_position_data["session_number"] = session_number
            processed_position_data["mouse_id"] = mouse_id
            processed_position_data["session_id"] = session_id

            all_behaviour = pd.concat([all_behaviour, processed_position_data], ignore_index=True)
            print("successfully processed and saved "+recording)

        except Exception as ex:
            print('This is what Python says happened:')
            print(ex)
            exc_type, exc_value, exc_traceback = sys.exc_info()
            traceback.print_tb(exc_traceback)
            print("couldn't process vr_grid analysis on "+recording)

    return all_behaviour


def main():

    recording_paths = []
    recording_paths.extend([f.path for f in os.scandir("/mnt/datastore/Harry/Cohort11_april2024/vr") if f.is_dir()])
    #all_behaviour = process_recordings(recording_paths)
    #all_behaviour.to_pickle("/mnt/datastore/Harry/Cohort11_april2024/summary/all_behaviour.pkl")
    all_behaviour = pd.read_pickle("/mnt/datastore/Harry/Cohort11_april2024/summary/all_behaviour.pkl")

    for mouse in np.unique(all_behaviour["mouse_id"]):
        plot_vr_stop_hists(all_behaviour[all_behaviour["mouse_id"] == mouse], save_path="/mnt/datastore/Harry/Cohort11_april2024/summary/",title=mouse, track_length=200)
        plot_vr_stop_rasters(all_behaviour[all_behaviour["mouse_id"] == mouse], save_path="/mnt/datastore/Harry/Cohort11_april2024/summary/",title=mouse, track_length=200)

if __name__ == '__main__':
    main()