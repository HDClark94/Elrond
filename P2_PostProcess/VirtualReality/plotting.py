import os
import matplotlib.pylab as plt
import numpy as np
from scipy import stats
import matplotlib.ticker as ticker
import settings
from Helpers import plot_utility
from Helpers.array_utility import pandas_collumn_to_numpy_array, pandas_collumn_to_2d_numpy_array

def plot_eye(processed_position_data, output_path="", track_length=200):
    save_path = output_path+'/Figures/behaviour'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    x_max = len(processed_position_data)
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
    trial_radi = pandas_collumn_to_2d_numpy_array(processed_position_data["radi_binned_in_space"])

    # remove values that aren't consisent with the radius of the eye changes, changes by a factor of 10 are insane
    median_val = np.nanmedian(trial_radi) # use this as a reference for a sensible radius picked up
    trial_radi[trial_radi > median_val*10] = np.nan
    trial_radi[trial_radi < median_val/10] = np.nan

    where_are_NaNs = np.isnan(trial_radi)
    #trial_radi[where_are_NaNs] = 0
    locations = np.arange(0, len(trial_radi[0]))
    ordered = np.arange(0, len(trial_radi), 1)
    X, Y = np.meshgrid(locations, ordered)
    cmap = plt.cm.get_cmap("cool")
    cmap.set_bad("white")
    pcm = ax.pcolormesh(X, Y, trial_radi, cmap=cmap, shading="auto")
    cbar = fig.colorbar(pcm, ax=ax, fraction=0.046, pad=0.14)
    cbar.mappable.set_clim(0, np.nanpercentile(trial_radi, 99))
    cbar.outline.set_visible(False)
    #cbar.set_ticks([0,100])
    #cbar.set_ticklabels(["0", "100"])
    cbar.ax.tick_params(labelsize=20)
    cbar.set_label('Eye radius (pixels)', fontsize=20, rotation=270)
    plt.ylabel('Trial Number', fontsize=25, labelpad = 10)
    plt.xlabel('Location (cm)', fontsize=25, labelpad = 10)
    plt.xlim(0, track_length)
    plt.ylim(0, len(processed_position_data))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(100))
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plot_utility.style_vr_plot(ax, x_max)
    plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.2, right = 0.87, top = 0.92)
    plt.savefig(save_path + '/eye_heat_map.png', dpi=200)
    plt.close()

    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
    trial_radi = pandas_collumn_to_2d_numpy_array(processed_position_data["radi_binned_in_space"])
    trial_radi_sem = stats.sem(trial_radi, axis=0, nan_policy="omit")
    trial_radi_avg = np.nanmean(trial_radi, axis=0)
    bin_centres = np.array(processed_position_data["position_bin_centres"].iloc[0])
    ax.fill_between(bin_centres, trial_radi_avg-trial_radi_sem, trial_radi_avg+trial_radi_sem, color="black", alpha=0.2)
    ax.plot(bin_centres, trial_radi_avg, color="black", linewidth=3)
    plt.xlim(0,track_length)
    #ax.set_yticks([0, 50, 100])
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plot_utility.style_track_plot(ax, track_length)
    plt.ylabel('Eye radius (pixels)', fontsize=25, labelpad = 10)
    plt.xlabel('Location (cm)', fontsize=25, labelpad = 10)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(100))
    #ax.yaxis.set_major_locator(ticker.MultipleLocator(100))
    ax.tick_params(axis='both', which='major', labelsize=20)
    plot_utility.style_vr_plot(ax, x_max=None)
    plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.2, right = 0.87, top = 0.92)
    plt.savefig(save_path + '/eye_radius_vs_track_position.png', dpi=200)
    plt.close()



def plot_speed_heat_map(processed_position_data, output_path="", track_length=200):
    save_path = output_path+'/Figures/behaviour'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    x_max = len(processed_position_data)
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
    trial_speeds = pandas_collumn_to_2d_numpy_array(processed_position_data["speeds_binned_in_space"])
    where_are_NaNs = np.isnan(trial_speeds)
    #trial_speeds[where_are_NaNs] = 0
    locations = np.arange(0, len(trial_speeds[0]))
    ordered = np.arange(0, len(trial_speeds), 1)
    X, Y = np.meshgrid(locations, ordered)
    cmap = plt.cm.get_cmap("jet")
    cmap.set_bad("white")
    pcm = ax.pcolormesh(X, Y, trial_speeds, cmap=cmap, shading="auto")
    cbar = fig.colorbar(pcm, ax=ax, fraction=0.046, pad=0.14)
    cbar.mappable.set_clim(0, 100)
    cbar.outline.set_visible(False)
    cbar.set_ticks([0,100])
    cbar.set_ticklabels(["0", "100"])
    cbar.ax.tick_params(labelsize=20)
    cbar.set_label('Speed (cm/s)', fontsize=20, rotation=270)
    plt.ylabel('Trial Number', fontsize=25, labelpad = 10)
    plt.xlabel('Location (cm)', fontsize=25, labelpad = 10)
    plt.xlim(0, track_length)
    plt.ylim(0, len(processed_position_data))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(100))
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plot_utility.style_vr_plot(ax, x_max)
    plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.2, right = 0.87, top = 0.92)
    plt.savefig(save_path + '/speed_heat_map.png', dpi=200)
    plt.close()



# plot the raw movement channel to check all is good
def plot_movement_channel(location, output_path):
    save_path = output_path + '/Figures'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    plt.plot(location)
    plt.savefig(save_path + '/movement' + '.png')
    plt.close()

# plot the trials to check all is good
def plot_trials(trials, output_path):
    save_path = output_path + '/Figures'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    plt.plot(trials)
    plt.savefig(save_path + '/trials' + '.png')
    plt.close()

def plot_velocity(velocity, output_path):
    save_path = output_path + '/Figures'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    plt.plot(velocity)
    plt.savefig(save_path + '/velocity' + '.png')
    plt.close()

def plot_running_mean_velocity(velocity, output_path):
    save_path = output_path + '/Figures'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    plt.plot(velocity)
    plt.savefig(save_path + '/running_mean_velocity' + '.png')
    plt.close()

# plot the raw trial channels to check all is good
def plot_trial_channels(trial1, trial2, output_path):
    plt.plot(trial1[0,:])
    plt.savefig(output_path + '/Figures/trial_type1.png')
    plt.close()
    plt.plot(trial2[0,:])
    plt.savefig(output_path + '/Figures/trial_type2.png')
    plt.close()


'''

# Plot behavioural info:
> stops on trials 
> avg stop histogram
> avg speed histogram
> combined plot

'''

def get_trial_color(trial_type):
    if trial_type == 0:
        return "black"
    elif trial_type == 1:
        return "red"
    elif trial_type == 2:
        return "blue"
    else:
        print("invalid trial-type passed to get_trial_color()")

def plot_stops_on_track(processed_position_data, output_path, track_length=200):
    print('I am plotting stop rasta...')
    save_path = output_path+'/Figures/behaviour'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)

    for index, trial_row in processed_position_data.iterrows():
        trial_row = trial_row.to_frame().T.reset_index(drop=True)
        trial_type = trial_row["trial_type"].iloc[0]
        trial_number = trial_row["trial_number"].iloc[0]
        trial_stop_color = get_trial_color(trial_type)

        ax.plot(np.array(trial_row["stop_location_cm"].iloc[0]),
                trial_number*np.ones(len(trial_row["stop_location_cm"].iloc[0])),
                'o', color=trial_stop_color, markersize=4)

    plt.ylabel('Stops on trials', fontsize=25, labelpad = 10)
    plt.xlabel('Location (cm)', fontsize=25, labelpad = 10)
    plt.xlim(0,track_length)
    ax.tick_params(axis='both', which='major', labelsize=20)
    plt.ylim(0,len(processed_position_data))
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plot_utility.style_track_plot(ax, track_length)
    n_trials = len(processed_position_data)
    x_max = n_trials+0.5
    plot_utility.style_vr_plot(ax, x_max)
    plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.2, right = 0.87, top = 0.92)
    plt.savefig(save_path + '/stop_raster.png', dpi=200)
    plt.close()

def plot_variables(position_data, output_path): # can be raw or downsampled
    save_path = output_path+'/Figures/behaviour'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    for column in list(position_data):
        print("plotting", column)
        print("avg value:", str(np.nanmean(position_data[column])))
        plt.plot(position_data[column])
        plt.savefig(save_path + '/' + column + '.png')
        plt.close()

def curate_stops(stop_locations, stop_trial_numbers, track_length):
    stop_locations = np.array(stop_locations)
    stop_trial_numbers = np.array(stop_trial_numbers)
    stop_locations_elapsed = (track_length * (stop_trial_numbers - 1)) + stop_locations

    curated_stop_location_elapsed = []
    curated_stop_locations = []
    curated_stop_trials = []
    for i, stop_loc in enumerate(stop_locations_elapsed):
        if (i == 0):  # take first stop always
            add_stop = True
        elif ((stop_locations_elapsed[i] - max(curated_stop_location_elapsed)) > 1):
            # only include stop if the stop was at least 1cm away for a curated stop
            add_stop = True
        else:
            add_stop = False

        if add_stop:
            curated_stop_location_elapsed.append(stop_locations_elapsed[i])
            curated_stop_locations.append(stop_locations[i])
            curated_stop_trials.append(stop_trial_numbers[i])

    return np.array(curated_stop_locations), np.array(curated_stop_trials)


def plot_stop_histogram(processed_position_data, output_path="", track_length=200):
    # TODO test this
    print('plotting stop histogram...')
    save_path = output_path+'/Figures/behaviour'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(1, 1, 1)
    bin_size = 5

    for tt, c in zip([0,1,2], ["black", "red", "blue"]):
        tt_processed_position_data = processed_position_data[processed_position_data["trial_type"] == tt]

        tt_stops = []
        tt_trial_numbers = []
        for i, tn in enumerate(tt_processed_position_data["trial_number"]):
            tt_stops.extend(tt_processed_position_data["stop_location_cm"].iloc[i])
            tt_trial_numbers.extend(np.ones(len(tt_processed_position_data["stop_location_cm"].iloc[i]))*tn)

        tt_stops, tt_trial_numbers = curate_stops(tt_stops, tt_trial_numbers, track_length)
        tt_stops_hist, bin_edges = np.histogram(tt_stops, bins=int(track_length/bin_size), range=(0, track_length))
        bin_centres = 0.5 * (bin_edges[1:] + bin_edges[:-1])
        if len(tt_processed_position_data) > 0:
            ax.plot(bin_centres, tt_stops_hist / len(tt_processed_position_data), '-', color=c)

    plt.ylabel('Stops/Trial', fontsize=25, labelpad = 10)
    plt.xlabel('Location (cm)', fontsize=25, labelpad = 10)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.tick_params(axis='both', which='major', labelsize=20)
    plt.xlim(0,track_length)
    ax.set_ylim(bottom=0)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plot_utility.style_track_plot(ax, track_length)
    plot_utility.style_vr_plot(ax)
    plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.2, right = 0.87, top = 0.92)
    plt.savefig(save_path + '/stop_histogram.png', dpi=200)
    plt.close()


def min_max_normalize(x):
    """
        argument
            - x: input image data in numpy array [32, 32, 3]
        return
            - normalized x
    """
    min_val = np.min(x)
    max_val = np.max(x)
    x = (x-min_val) / (max_val-min_val)
    return x


def plot_speed_histogram(processed_position_data, output_path="", track_length=200):
    # TODO test this
    save_path = output_path + '/Figures/behaviour'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
    for tt, c in zip([0,1,2], ["black", "red", "blue"]):
        tt_processed_position_data = processed_position_data[processed_position_data["trial_type"] == tt]
        if len(tt_processed_position_data)>0:
            trial_speeds = pandas_collumn_to_2d_numpy_array(tt_processed_position_data["speeds_binned_in_space"])
            trial_speeds_sem = stats.sem(trial_speeds, axis=0, nan_policy="omit")
            trial_speeds_avg = np.nanmean(trial_speeds, axis=0)
            bin_centres = np.array(processed_position_data["position_bin_centres"].iloc[0])
            ax.plot(bin_centres, trial_speeds_avg, color=c, linewidth=4)

    plt.xlim(0,track_length)
    ax.set_yticks([0, 50, 100])
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plot_utility.style_track_plot(ax, track_length)
    plt.ylabel('Speed (cm/s)', fontsize=25, labelpad = 10)
    plt.xlabel('Location (cm)', fontsize=25, labelpad = 10)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(100))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(100))
    ax.tick_params(axis='both', which='major', labelsize=20)
    plot_utility.style_vr_plot(ax, x_max=115)
    plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.2, right = 0.87, top = 0.92)
    plt.savefig(save_path + '/speed_histogram.png', dpi=200)
    plt.close()




def plot_spikes_on_track(spike_data, processed_position_data, output_path, track_length=200,
                         plot_trials=["beaconed", "non_beaconed", "probe"]):
    print('plotting spike rastas...')
    save_path = output_path + '/Figures/spike_trajectories'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        cluster_spike_data = spike_data[spike_data["cluster_id"] == cluster_id]
        firing_times_cluster = cluster_spike_data.firing_times.iloc[0]
        if len(firing_times_cluster)>1:

            x_max = len(processed_position_data)+1
            if x_max>100:
                spikes_on_track = plt.figure(figsize=(4,(x_max/32)))
            else:
                spikes_on_track = plt.figure(figsize=(4,(x_max/20)))

            ax = spikes_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)

            if "beaconed" in plot_trials:
                ax.plot(cluster_spike_data.iloc[0].beaconed_position_cm, cluster_spike_data.iloc[0].beaconed_trial_number, '|', color='Black', markersize=4)
            if "non_beaconed" in plot_trials:
                ax.plot(cluster_spike_data.iloc[0].nonbeaconed_position_cm, cluster_spike_data.iloc[0].nonbeaconed_trial_number, '|', color='Red', markersize=4)
            if "probe" in plot_trials:
                ax.plot(cluster_spike_data.iloc[0].probe_position_cm, cluster_spike_data.iloc[0].probe_trial_number, '|', color='Blue', markersize=4)

            plt.ylabel('Spikes on trials', fontsize=20, labelpad = 10)
            plt.xlabel('Location (cm)', fontsize=20, labelpad = 10)
            plt.xlim(0,track_length)
            ax.yaxis.set_ticks_position('left')
            ax.xaxis.set_ticks_position('bottom')

            plot_utility.style_track_plot(ax, track_length)
            plot_utility.style_vr_plot(ax, x_max)
            plt.locator_params(axis = 'y', nbins  = 4)
            try:
                plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
            except ValueError:
                continue
            if len(plot_trials)<3:
                plt.savefig(save_path + '/' + spike_data.session_id.iloc[cluster_index] + '_track_firing_Cluster_' + str(cluster_id) + "_" + str("_".join(plot_trials)) + '.png', dpi=200)
            else:
                plt.savefig(save_path + '/' + spike_data.session_id.iloc[cluster_index] + '_track_firing_Cluster_' + str(cluster_id) + '.png', dpi=200)
            plt.close()


def plot_firing_rate_maps(spike_data, processed_position_data, output_path, track_length=200):
    print('I am plotting firing rate maps...')
    save_path = output_path + '/Figures/spike_rate'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        cluster_spike_data = spike_data[spike_data["cluster_id"] == cluster_id]
        firing_times_cluster = np.array(cluster_spike_data["firing_times"].iloc[0])

        if len(firing_times_cluster)>1:
            if "fr_binned_in_space" in list(cluster_spike_data):
                fr_column = "fr_binned_in_space"
            elif "fr_binned_in_space_smoothed" in list(cluster_spike_data):
                fr_column = "fr_binned_in_space_smoothed"
            fr_binned_in_space = np.array(cluster_spike_data[fr_column].iloc[0])
            fr_binned_in_space_bin_centres = np.array(cluster_spike_data['fr_binned_in_space_bin_centres'].iloc[0])[0]

            spikes_on_track = plt.figure()
            spikes_on_track.set_size_inches(5, 5, forward=True)
            ax = spikes_on_track.add_subplot(1, 1, 1)
            plot_utility.style_track_plot(ax, track_length)
            y_max=0

            for tt, c in zip([0, 1, 2], ["black", "red", "blue"]):
                tt_processed_position_data = processed_position_data[processed_position_data["trial_type"] == tt]
                tt_trial_numbers = np.asarray(tt_processed_position_data["trial_number"])
                tt_fr_binned_in_space = fr_binned_in_space[tt_trial_numbers-1]
                ax.fill_between(fr_binned_in_space_bin_centres, np.nanmean(tt_fr_binned_in_space, axis=0)-stats.sem(tt_fr_binned_in_space, axis=0), np.nanmean(tt_fr_binned_in_space, axis=0)+stats.sem(tt_fr_binned_in_space, axis=0), color=c, alpha=0.3)
                ax.plot(fr_binned_in_space_bin_centres, np.nanmean(tt_fr_binned_in_space, axis=0), color=c)

                fr_max = max(np.nanmean(tt_fr_binned_in_space, axis=0)+stats.sem(tt_fr_binned_in_space, axis=0))
                y_max = max([y_max, fr_max])
                y_max = np.ceil(y_max)

            plt.ylabel('Firing Rate (Hz)', fontsize=20, labelpad = 20)
            plt.xlabel('Location (cm)', fontsize=20, labelpad = 20)
            plt.xlim(0, track_length)
            tick_spacing = 100
            ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
            ax.yaxis.set_ticks_position('left')
            ax.xaxis.set_ticks_position('bottom')
            plot_utility.style_vr_plot(ax, x_max=y_max)
            ax.set_yticks([0, np.round(ax.get_ylim()[1], 2)])
            plt.locator_params(axis = 'y', nbins  = 4)
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.tight_layout()
            plt.savefig(save_path + '/' + spike_data.session_id.iloc[cluster_index] + '_rate_map_Cluster_' + str(cluster_id) + '.png', dpi=300)
            plt.close()

    return spike_data


def plot_behaviour(position_data, processed_position_data, output_path, track_length):
    plot_variables(position_data, output_path)
    plot_stops_on_track(processed_position_data, output_path, track_length=track_length)
    plot_stop_histogram(processed_position_data, output_path, track_length=track_length)
    plot_speed_histogram(processed_position_data, output_path, track_length=track_length)
    plot_speed_heat_map(processed_position_data, output_path, track_length=track_length)
    plot_eye(processed_position_data, output_path, track_length=track_length)

def plot_track_firing(spike_data, processed_position_data, output_path, track_length):
    plot_firing_rate_maps(spike_data, processed_position_data, output_path, track_length=track_length)
    plot_spikes_on_track(spike_data, processed_position_data, output_path, track_length=track_length,
                         plot_trials=["beaconed", "non_beaconed", "probe"])


#  this is here for testing
def main():
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')


if __name__ == '__main__':
    main()


