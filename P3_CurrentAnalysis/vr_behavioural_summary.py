# plot summary data for a given mouse for the vr task
import pandas as pd
import os
import traceback
import sys
import numpy as np
import matplotlib.pyplot as plt
from P2_PostProcess.VirtualReality.plotting import *
from Helpers.array_utility import pandas_collumn_to_numpy_array
import scipy.stats as stats


def plot_speed_distibutions(position_data, save_path, title, track_length):
    fig, axs = plt.subplots(7, 7, figsize=(15, 10), sharex=True, sharey=False)
    fig.suptitle(title, fontsize=16)
 
    days = np.unique(position_data["session_number"])
    for idx, day in enumerate(days):
        ax = axs.flat[idx]
        day_position_data = position_data[position_data["session_number"] == day]
        column = "speed_as_read_by_blender"
        speeds = np.asarray(day_position_data[column])
        speeds = speeds[~np.isnan(speeds)] 
        ax.hist(speeds, density=True, bins=50, color="black")
        ax.axvline(np.nanmean(speeds), color="red")
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.set_ylim(bottom=0) 
        ax.set_xlim(left=-20, right=100)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        ax.set_title(f'Day {day}')

    # Set shared x and y labels
    fig.text(0.5, 0.04, 'Speed (cm/s)', ha='center', fontsize=18)
    fig.text(0.04, 0.5, 'Density', va='center', rotation='vertical', fontsize=18)
    # Adjust layout to prevent overlapping
    plt.tight_layout(rect=[0.03, 0.03, 1, 0.95])
    plt.subplots_adjust(left=0.1, bottom=0.1)
    plt.savefig(save_path + title + '_speed_distribution.png', dpi=200)
    return


def plot_vr_stop_hists(processed_position_data, save_path, title, track_length):
    fig, axs = plt.subplots(7, 7, figsize=(15, 10), sharex=True, sharey=False)
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

def get_tt_str(tt):
    if tt == 0:
        return "Beaconed"
    elif tt == 1:
        return "Non-Beaconed"
    elif tt == 2:
        return "Probe"
    else:
        return ""

def plot_vr_hits_across_mice(processed_position_data, save_path, track_length):
    mouse_colors = ['darkturquoise', 'salmon', u'#2ca02c', u'#d62728', u'#9467bd',
                    u'#8c564b', u'#e377c2', u'#7f7f7f', u'#bcbd22', u'#17becf']

    trial_types = np.unique(processed_position_data["trial_type"])
    mice = np.unique(processed_position_data["mouse_id"])
    days = np.unique(processed_position_data["session_number"])

    fig, axs = plt.subplots(1, len(trial_types), figsize=(10, 5), sharey=True)
    hits = np.zeros((len(mice), len(days), len(trial_types)))
    hits[:,:,:] = np.nan
    for mouse_idx, mouse in enumerate(mice):
        for day_idx, day in enumerate(days):
            for tt_idx, tt in enumerate(trial_types):
                ax = axs[tt_idx]
                processed = processed_position_data[(processed_position_data["mouse_id"] == mouse) &
                                                    (processed_position_data["session_number"] == day) &
                                                    (processed_position_data["trial_type"] == tt)]
                if len(processed)>0:
                    hits[mouse_idx, day_idx, tt_idx] = 100*(len(processed[processed["hit_miss_try"] == "hit"]))\
                                                       /len(processed)

    # plot
    for tt_idx, tt in enumerate(trial_types):
        ax = axs[tt_idx]
        ax.plot(days, np.nanmean(hits[:, :, tt_idx], axis=0), color="black", alpha=1)
        ax.errorbar(days, np.nanmean(hits[:, :, tt_idx], axis=0),
                    stats.sem(hits[:, :, tt_idx], axis=0, nan_policy="omit"),
                    capsize=5, color="black", alpha=1)

        for mouse_idx, mouse in enumerate(mice):
            ax.scatter(days, hits[mouse_idx, :, tt_idx], color=mouse_colors[mouse_idx], alpha=0.5)
            ax.plot(days, hits[mouse_idx, :, tt_idx], color=mouse_colors[mouse_idx], alpha=0.3)
        ax.set_title(get_tt_str(tt), fontsize=18)
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.set_ylim(0, 100)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')

    fig.text(0.5, 0.04, 'Session number', ha='center', fontsize=18)
    fig.text(0.04, 0.5, 'Hit (%)', va='center', rotation='vertical', fontsize=18)
    # Adjust layout to prevent overlapping
    plt.tight_layout(rect=[0.03, 0.03, 1, 0.95])
    plt.subplots_adjust(left=0.12, bottom=0.15)
    plt.savefig(save_path + '_hits.png', dpi=200)
    return


def plot_vr_stop_rasters(processed_position_data, save_path, title, track_length):
    fig, axs = plt.subplots(7, 7, figsize=(15, 10), sharex=True, sharey=False)
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

def process_recordings(vr_recording_path_list, derivatives_path):
    all_processed_position = pd.DataFrame()
    all_position = pd.DataFrame()

    for recording in vr_recording_path_list:
        print("processing ", recording)
        try:
            session_id = recording.split("/")[-1]
            mouse_id = session_id.split("_")[0]
            day_id = session_id.split("_")[1]
            session_number = int(session_id.split("_")[1].split("D")[-1])

            position_data = pd.read_csv(derivatives_path+mouse_id+"/"+day_id+"/vr/"+session_id+"/processed/position_data.csv")
            processed_position_data = pd.read_pickle(derivatives_path+mouse_id+"/"+day_id+"/vr/"+session_id+"/processed/processed_position_data.pkl")

            processed_position_data["session_number"] = session_number
            processed_position_data["mouse_id"] = mouse_id
            processed_position_data["session_id"] = session_id
            position_data["mouse_id"] = mouse_id
            position_data["session_id"] = session_id
            position_data["session_number"] = session_number

            all_processed_position = pd.concat([all_processed_position, processed_position_data], ignore_index=True)
            all_position = pd.concat([all_position, position_data], ignore_index=True)
            print("successfully processed and saved "+recording)

        except Exception as ex:
            print('This is what Python says happened:')
            print(ex)
            exc_type, exc_value, exc_traceback = sys.exc_info()
            traceback.print_tb(exc_traceback)
            print("couldn't process vr_grid analysis on "+recording)

    return all_processed_position, all_position


def main():
    cohort = "Cohort12_august2024"
    #cohort = "Cohort11_april2024" 
    recording_paths = [] 
    recording_paths.extend([f.path for f in os.scandir("/mnt/datastore/Harry/"+cohort+"/vr") if f.is_dir()])
    all_processed_position, all_position = process_recordings(recording_paths, derivatives_path="/mnt/datastore/Harry/"+cohort+"/derivatives/")
    all_processed_position.to_pickle("/mnt/datastore/Harry/"+cohort+"/summary/all_processed_position.pkl")
    all_position.to_pickle("/mnt/datastore/Harry/"+cohort+"/summary/all_position.pkl")
    
    all_processed_position = pd.read_pickle("/mnt/datastore/Harry/"+cohort+"/summary/all_processed_position.pkl")
    all_position = pd.read_pickle("/mnt/datastore/Harry/"+cohort+"/summary/all_position.pkl")

    plot_vr_hits_across_mice(all_processed_position,save_path="/mnt/datastore/Harry/"+cohort+"/summary/", track_length=200)
    for mouse in ["M22", "M26"]: 
        plot_speed_distibutions(all_position[all_position["mouse_id"] == mouse], save_path="/mnt/datastore/Harry/"+cohort+"/summary/",title=mouse, track_length=200)
        plot_vr_stop_hists(all_processed_position[all_processed_position["mouse_id"] == mouse], save_path="/mnt/datastore/Harry/"+cohort+"/summary/",title=mouse, track_length=200)
        plot_vr_stop_rasters(all_processed_position[all_processed_position["mouse_id"] == mouse], save_path="/mnt/datastore/Harry/"+cohort+"/summary/",title=mouse, track_length=200)
if __name__ == '__main__': 
    main()