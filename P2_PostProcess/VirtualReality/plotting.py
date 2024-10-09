import os
import matplotlib.pylab as plt
import matplotlib.image as mpimg
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.ticker as ticker
import settings as settings
from astropy.nddata import block_reduce
from Helpers import plot_utility
from Helpers.array_utility import pandas_collumn_to_2d_numpy_array
from astropy.convolution import convolve, Gaussian1DKernel
from P3_CurrentAnalysis.basic_lomb_scargle_estimator import lomb_scargle, distance_from_integer, frequency

def plot_eye_trajectory(position_data, processed_position_data, output_path="", track_length=200):
    save_path = output_path+'Figures/behaviour'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    fig, axs = plt.subplots(2, 3, figsize=(10, 7))
    #axs[0].set_ylabel('Eye radius (pixels)', fontsize=25, labelpad = 10)
    y_max = 0; y_min = 1e15  
    x_max = 0; x_min = 1e15
    for tt, ax_tt in zip([0, 1], axs):
        for hmt, hmt_color, ax in zip(["hit", "try", "run"], ["green", "orange", "red"], ax_tt):
            tmp_processed_position_data = processed_position_data[(processed_position_data["hit_miss_try"] == hmt) &
                                                                  (processed_position_data["trial_type"] == tt)]
            trial_numbers = np.unique(tmp_processed_position_data["trial_number"])

            tmp_position_data = position_data[np.isin(position_data["trial_number"], trial_numbers)]

            if len(tmp_position_data)>0:        
                # Bin the track_location in 1cm bins
                tmp_position_data['track_bin'] = pd.cut(tmp_position_data['x_position_cm'], bins=np.arange(0, tmp_position_data['x_position_cm'].max() + 1, 1))
                # Group by trial_number and track_bin, then calculate the mean x_pos and y_pos
                grouped = tmp_position_data.groupby(['trial_number', 'track_bin']).agg({'eye_centroid_x': 'mean', 'eye_centroid_y': 'mean'}).reset_index()
                # Group by track_bin to average across trials
                averaged_data = grouped.groupby('track_bin').agg({'eye_centroid_x': 'mean', 'eye_centroid_y': 'mean'}).reset_index()
                colors = plt.cm.rainbow(np.linspace(0, 1, len(averaged_data['eye_centroid_y'])))
                ax.scatter(averaged_data['eye_centroid_x'], averaged_data['eye_centroid_y'], marker='o', alpha=1, color=colors)

                """
                for tn in trial_numbers:
                    colors = plt.cm.rainbow(np.linspace(0, 1, len(tmp_position_data[tmp_position_data["trial_number"] == tn])))
                    ax.scatter(tmp_position_data[tmp_position_data["trial_number"] == tn]["eye_centroid_x"],
                            tmp_position_data[tmp_position_data["trial_number"] == tn]["eye_centroid_y"], color=colors, alpha=0.1)
                """ 
                if y_max < np.max(averaged_data["eye_centroid_y"]):
                    y_max = np.max(averaged_data["eye_centroid_y"])
                if y_min > np.min(averaged_data["eye_centroid_y"]):
                    y_min = np.min(averaged_data["eye_centroid_y"])
                if x_max < np.max(averaged_data["eye_centroid_x"]):
                    x_max = np.max(averaged_data["eye_centroid_x"])
                if x_min > np.min(averaged_data["eye_centroid_x"]):
                    x_min = np.min(averaged_data["eye_centroid_x"])
            ax.set_xlim(0,track_length)
            ax.set_title(f'tt:{tt},  hmt:{hmt}')
            ax.yaxis.set_ticks_position('left')
            ax.xaxis.set_ticks_position('bottom')
            ax.xaxis.set_major_locator(ticker.MultipleLocator(100))
            ax.tick_params(axis='both', which='major', labelsize=20)
            plot_utility.style_vr_plot(ax, x_max=None)
    for ax in axs.flatten():
        ax.set_ylim(y_min,y_max) 
        ax.set_xlim(x_min,x_max)
        ax.set_xticks([])
        ax.set_yticks([]) 
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
    # Create a ScalarMappable and add a colorbar
    sm = plt.cm.ScalarMappable(cmap='rainbow', norm=plt.Normalize(vmin=0, vmax=len(averaged_data['eye_centroid_y'])))
    sm.set_array([])  # Only needed for the colorbar
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('(cm)')
    plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.2, right = 0.87, top = 0.92)
    plt.savefig(save_path + '/eye_movements.png', dpi=200)
    plt.close()

def plot_eye(processed_position_data, output_path="", track_length=200):
    save_path = output_path+'Figures/behaviour'
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

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].set_ylabel('Eye radius (pixels)', fontsize=25, labelpad = 10)
    y_max = 0
    y_min = 1e15
    for tt, ax in zip([0, 1], axs):
        for hmt, hmt_color in zip(["hit", "try", "run"], ["green", "orange", "red"]):
            tmp_processed_position_data = processed_position_data[(processed_position_data["hit_miss_try"] == hmt) &
                                                                  (processed_position_data["trial_type"] == tt)]
            if len(tmp_processed_position_data)>0: 
                trial_radi = pandas_collumn_to_2d_numpy_array(tmp_processed_position_data["radi_binned_in_space"])
                trial_radi_sem = stats.sem(trial_radi, axis=0, nan_policy="omit")
                trial_radi_avg = np.nanmean(trial_radi, axis=0)
                bin_centres = np.array(processed_position_data["position_bin_centres"].iloc[0])
                ax.fill_between(bin_centres, trial_radi_avg-trial_radi_sem, trial_radi_avg+trial_radi_sem, color=hmt_color, alpha=0.2)
                ax.plot(bin_centres, trial_radi_avg, color=hmt_color, linewidth=3)

            if y_max < np.max(trial_radi_avg+trial_radi_sem):
                y_max = np.max(trial_radi_avg+trial_radi_sem)
            if y_min > np.min(trial_radi_avg-trial_radi_sem):
                y_min = np.min(trial_radi_avg-trial_radi_sem)
            ax.set_xlim(0,track_length)

            ax.yaxis.set_ticks_position('left')
            ax.xaxis.set_ticks_position('bottom')
            ax.set_xlabel('Location (cm)', fontsize=25, labelpad = 10)
            ax.xaxis.set_major_locator(ticker.MultipleLocator(100))
            ax.tick_params(axis='both', which='major', labelsize=20)
            plot_utility.style_vr_plot(ax, x_max=None)
        if tt == 0:
            plot_utility.style_track_plot(ax, track_length)
        else:
            plot_utility.style_track_plot_no_cue(ax, track_length)
    axs[0].set_ylim(y_min,y_max) 
    axs[1].set_ylim(y_min,y_max) 

    plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.2, right = 0.87, top = 0.92)
    plt.savefig(save_path + '/eye_radius_vs_track_position_trial_seperated.png', dpi=200)
    plt.close() 

def plot_speed_heat_map(processed_position_data, output_path="", track_length=200):
    save_path = output_path+'Figures/behaviour'
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
    pcm = ax.pcolormesh(X, Y, trial_speeds, cmap=cmap, shading="auto", vmin=0, vmax=np.nanmax(trial_speeds))
    #cbar = fig.colorbar(pcm, ax=ax, fraction=0.046, pad=0.14) 
    #cbar.mappable.set_clim(0, 100)
    #cbar.outline.set_visible(False)
    #cbar.set_ticks([0,100])
    #cbar.set_ticklabels(["0", "100"])
    #cbar.ax.tick_params(labelsize=20)
    #cbar.set_label('Speed (cm/s)', fontsize=20, rotation=270)
    plt.ylabel('Trial Number', fontsize=25, labelpad = 10)
    plt.xlabel('Location (cm)', fontsize=25, labelpad = 10)
    plt.xlim(0, track_length)
    plt.ylim(0, len(processed_position_data))
    ax.xaxis.set_major_locator(ticker.MultipleLocator(100))
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plot_utility.style_vr_plot(ax, x_max)
    plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.2, right = 0.87, top = 0.92)
    plt.savefig(save_path + '/speed_heat_map.png', dpi=200)
    plt.close()

# plot the raw movement channel to check all is good
def plot_movement_channel(location, output_path):
    save_path = output_path + 'Figures'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    plt.plot(location)
    plt.savefig(save_path + '/movement' + '.png')
    plt.close()

# plot the trials to check all is good
def plot_trials(trials, output_path):
    save_path = output_path + 'Figures'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    plt.plot(trials)
    plt.savefig(save_path + '/trials' + '.png')
    plt.close()

def plot_velocity(velocity, output_path):
    save_path = output_path + 'Figures'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    plt.plot(velocity)
    plt.savefig(save_path + '/velocity' + '.png')
    plt.close()

def plot_running_mean_velocity(velocity, output_path):
    save_path = output_path + 'Figures'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    plt.plot(velocity)
    plt.savefig(save_path + '/running_mean_velocity' + '.png')
    plt.close()

# plot the raw trial channels to check all is good
def plot_trial_channels(trial1, trial2, output_path):
    plt.plot(trial1[0,:])
    plt.savefig(output_path + 'Figures/trial_type1.png')
    plt.close()
    plt.plot(trial2[0,:])
    plt.savefig(output_path + 'Figures/trial_type2.png')
    plt.close()

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
    save_path = output_path+'Figures/behaviour'
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
    ax.xaxis.set_major_locator(ticker.MultipleLocator(100))
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
    save_path = output_path+'Figures/behaviour'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path) 

    for column in list(position_data):
        variables = np.asarray(position_data[column], dtype=np.float128)
        fig = plt.figure(figsize=(6,6))
        ax = fig.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
        ax.plot(variables, color="black")
        plt.savefig(save_path + '/' + column + '.png')
        plt.close()

        fig = plt.figure(figsize=(6,6))
        ax = fig.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
        ax.hist(variables, density=True, bins=50, color="black")
        ax.set_ylabel('Density', fontsize=25, labelpad = 10)
        ax.set_xlabel(column, fontsize=25, labelpad = 10)
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.set_ylim(bottom=0)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.2, right = 0.87, top = 0.92)
        plt.savefig(save_path + '/' + column + '_hist.png')
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
    save_path = output_path+'Figures/behaviour'
    print("saveing at...", save_path)
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
            ax.plot(bin_centres, tt_stops_hist / len(tt_processed_position_data), '-', color=c, linewidth=4)

    plt.ylabel('Stops/Trial', fontsize=25, labelpad = 10)
    plt.xlabel('Location (cm)', fontsize=25, labelpad = 10)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(100))
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
    save_path = output_path + 'Figures/behaviour'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
    y_max=0
    for tt, c in zip([0,1,2], ["black", "red", "blue"]):
        tt_processed_position_data = processed_position_data[processed_position_data["trial_type"] == tt]
        if len(tt_processed_position_data)>0:
            trial_speeds = pandas_collumn_to_2d_numpy_array(tt_processed_position_data["speeds_binned_in_space"])
            trial_speeds_sem = stats.sem(trial_speeds, axis=0, nan_policy="omit")
            trial_speeds_avg = np.nanmean(trial_speeds, axis=0)
            bin_centres = np.array(processed_position_data["position_bin_centres"].iloc[0])
            ax.plot(bin_centres, trial_speeds_avg, color=c, linewidth=4)
            ax.fill_between(bin_centres, trial_speeds_avg-trial_speeds_sem, 
                            trial_speeds_avg+trial_speeds_sem, color=c, alpha=0.3)
            if np.nanmax(trial_speeds_avg) > y_max:
                y_max = np.nanmax(trial_speeds_avg) 

    plt.xlim(0,track_length)
    plt.ylim(0,y_max+10)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plot_utility.style_track_plot(ax, track_length)
    plt.ylabel('Speed (cm/s)', fontsize=25, labelpad = 10)
    plt.xlabel('Location (cm)', fontsize=25, labelpad = 10)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(100))
    ax.tick_params(axis='both', which='major', labelsize=20)
    plot_utility.style_vr_plot(ax, x_max=y_max+10)
    plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.2, right = 0.87, top = 0.92)
    plt.savefig(save_path + '/speed_histogram.png', dpi=200)
    plt.close()

def plot_spikes_on_track(spike_data, processed_position_data, output_path, track_length=200,
                         plot_trials=["beaconed", "non_beaconed", "probe"]):
    print('plotting spike rastas...')
    save_path = output_path + 'Figures/spike_trajectories'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        cluster_spike_data = spike_data[spike_data["cluster_id"] == cluster_id]
        firing_times_cluster = cluster_spike_data.firing_times.iloc[0]
        x_position_cm = np.array(cluster_spike_data.x_position_cm.iloc[0])
        trial_numbers = np.array(cluster_spike_data.trial_number.iloc[0])
        trial_types = np.array(cluster_spike_data.trial_type.iloc[0]) 

        if len(firing_times_cluster)>1:
            x_max = len(processed_position_data)+1
            if x_max>100:
                spikes_on_track = plt.figure(figsize=(4,(x_max/32)))
            else:
                spikes_on_track = plt.figure(figsize=(4,(x_max/20)))

            ax = spikes_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
            if "beaconed" in plot_trials:
                ax.plot(x_position_cm[trial_types==0], trial_numbers[trial_types==0], '|', color='Black', markersize=4)
            if "non_beaconed" in plot_trials:
                ax.plot(x_position_cm[trial_types==1], trial_numbers[trial_types==1], '|', color='Red', markersize=4)
            if "probe" in plot_trials:
                ax.plot(x_position_cm[trial_types==2], trial_numbers[trial_types==2], '|', color='Blue', markersize=4)
            plt.ylabel('Spikes on trials', fontsize=20, labelpad = 10)
            plt.xlabel('Location (cm)', fontsize=20, labelpad = 10)
            plt.xlim(0,track_length)
            ax.yaxis.set_ticks_position('left')
            ax.xaxis.set_ticks_position('bottom')

            plot_utility.style_track_plot(ax, track_length)
            plot_utility.style_vr_plot(ax, x_max)
            plt.locator_params(axis = 'y', nbins  = 4)
            plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
            if len(plot_trials)<3:
                plt.savefig(save_path + '/' + spike_data.session_id.iloc[cluster_index] + '_track_firing_Cluster_' + str(cluster_id) + "_" + str("_".join(plot_trials)) + '.png', dpi=200)
            else:
                plt.savefig(save_path + '/' + spike_data.session_id.iloc[cluster_index] + '_track_firing_Cluster_' + str(cluster_id) + '.png', dpi=200)
            plt.close()
    return

def plot_firing_rate_maps_short(cluster_data, track_length=200, save_path=None):
    firing_times_cluster = cluster_data["firing_times"].iloc[0]
    cluster_id = cluster_data["cluster_id"].iloc[0]

    if len(firing_times_cluster)>1:
        cluster_firing_maps = np.array(cluster_data['fr_binned_in_space_smoothed'].iloc[0])
        cluster_firing_maps[np.isnan(cluster_firing_maps)] = np.nan
        cluster_firing_maps[np.isinf(cluster_firing_maps)] = np.nan

        spikes_on_track = plt.figure()
        spikes_on_track.set_size_inches(5, 5/3, forward=True)
        ax = spikes_on_track.add_subplot(1, 1, 1)
        locations = np.arange(0, len(cluster_firing_maps[0]))
        ax.fill_between(locations, np.nanmean(cluster_firing_maps, axis=0) - stats.sem(cluster_firing_maps, axis=0,nan_policy="omit"),
                                    np.nanmean(cluster_firing_maps, axis=0) + stats.sem(cluster_firing_maps, axis=0,nan_policy="omit"), color="black", alpha=0.2)
        ax.plot(locations, np.nanmean(cluster_firing_maps, axis=0), color="black", linewidth=1)
        
        plt.ylabel('FR (Hz)', fontsize=25, labelpad = 10)
        plt.xlabel('Location (cm)', fontsize=25, labelpad = 10)
        plt.xlim(0, track_length)
        ax.tick_params(axis='both', which='both', labelsize=20)
        ax.set_xlim([0, track_length])
        max_fr = max(np.nanmean(cluster_firing_maps, axis=0)+stats.sem(cluster_firing_maps, axis=0))
        max_fr = max_fr+(0.1*(max_fr))
        ax.set_ylim([0, max_fr])
        ax.set_yticks([0, np.round(ax.get_ylim()[1], 1)])
        ax.set_ylim(bottom=0)
        plot_utility.style_track_plot(ax, track_length, alpha=0.15)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(100))
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.3, right = 0.87, top = 0.92)
        if save_path is not None:
            plt.savefig(save_path + '/avg_firing_rate_maps_short_' + cluster_data.session_id.iloc[0] + '_' + str(int(cluster_id)) + '.png', dpi=300)
    return              

def plot_firing_rate_maps(spike_data, processed_position_data, output_path, track_length=200):
    print('I am plotting firing rate maps...')
    save_path = output_path + 'Figures/spike_rate'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    rate_maps_beaconed = []
    rate_maps_non_beaconed = []
    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        cluster_spike_data = spike_data[spike_data["cluster_id"] == cluster_id]
        firing_times_cluster = np.array(cluster_spike_data["firing_times"].iloc[0])

        if len(firing_times_cluster)>1:
            if "fr_binned_in_space_smoothed" in list(cluster_spike_data):
                fr_column = "fr_binned_in_space_smoothed"
            elif "fr_binned_in_space" in list(cluster_spike_data):
                fr_column = "fr_binned_in_space"
            fr_binned_in_space = np.array(cluster_spike_data[fr_column].iloc[0])
            fr_binned_in_space_bin_centres = np.array(cluster_spike_data['fr_binned_in_space_bin_centres'].iloc[0])[0]
            fr_binned_in_space[np.isnan(fr_binned_in_space)] = 0
            fr_binned_in_space[np.isinf(fr_binned_in_space)] = 0

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

                if tt == 0:
                    rate_maps_beaconed.append(np.nanmean(tt_fr_binned_in_space, axis=0))
                elif tt == 1:
                    rate_maps_non_beaconed.append(np.nanmean(tt_fr_binned_in_space, axis=0))

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

    gauss_kernel = Gaussian1DKernel(settings.guassian_std_for_smoothing_in_space_cm/settings.vr_bin_size_cm)
    nrows = int(np.ceil(np.sqrt(len(spike_data))))
    ncols = nrows; i=0; j=0;
    fig, ax = plt.subplots(ncols=ncols, nrows=nrows, figsize=(20, 20), squeeze=False)
    for rate_map_b, rate_map_nb in zip(rate_maps_beaconed, rate_maps_non_beaconed):
        rate_map_b = convolve(rate_map_b, gauss_kernel, boundary="extend")
        rate_map_nb = convolve(rate_map_nb, gauss_kernel, boundary="extend")
        ax[j, i].plot(fr_binned_in_space_bin_centres, rate_map_b, color="black")
        ax[j, i].plot(fr_binned_in_space_bin_centres, rate_map_nb, color="red")
        i+=1
        if i==ncols:
            i=0; j+=1
    for j in range(nrows):
        for i in range(ncols):
            plot_utility.style_track_plot(ax[j, i], track_length)
            plot_utility.style_vr_plot(ax[j, i])
            ax[j, i].spines['top'].set_visible(False)
            ax[j, i].spines['right'].set_visible(False)
            ax[j, i].spines['bottom'].set_visible(False)
            ax[j, i].spines['left'].set_visible(False)
            ax[j, i].set_xticks([])
            ax[j, i].set_yticks([])
            ax[j, i].xaxis.set_tick_params(labelbottom=False)
            ax[j, i].yaxis.set_tick_params(labelleft=False)
    plt.subplots_adjust(hspace=.1, wspace=.1, bottom=None, left=None, right=None, top=None)
    plt.savefig(save_path + '/all_firing_rates.png', dpi=400)
    plt.close()

def plot_spatial_periodogram_per_trial(spike_data, processed_position_data, output_path, track_length): 
    if "ls_powers" not in list(spike_data): 
        spike_data = lomb_scargle(spike_data, processed_position_data, track_length) 

    save_path = output_path + 'Figures/spatial_periodograms/'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path) 

    periodograms = []
    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        cluster_spike_data = spike_data[spike_data["cluster_id"] == cluster_id]
        firing_times_cluster = np.array(cluster_spike_data["firing_times"].iloc[0])

        if len(firing_times_cluster)>1:
            powers = np.array(cluster_spike_data["ls_powers"].iloc[0])
            centre_xs = np.array(cluster_spike_data["ls_centre_distances"].iloc[0])
            #centre_xs = np.round(centre_trials).astype(np.int64)

            fig = plt.figure()
            fig.set_size_inches(5, 5, forward=True)
            ax = fig.add_subplot(1, 1, 1)
            Y, X = np.meshgrid(centre_xs, frequency) 
            cmap = plt.cm.get_cmap("inferno")
            ax.pcolormesh(X, Y, powers.T, cmap=cmap)
            for f in range(1,5):
                ax.axvline(x=f, color="white", linewidth=2,linestyle="dotted")
            ax.set_ylabel('Centre X', fontsize=30, labelpad = 10)
            ax.set_xlabel('Track frequency', fontsize=30, labelpad = 10)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_xticks([0, 1, 2, 3, 4, 5])
            ax.set_xlim([0.1,5])
            ax.set_ylim([min(centre_xs), max(centre_xs)])
            ax.yaxis.set_tick_params(labelsize=20)
            ax.xaxis.set_tick_params(labelsize=20)
            plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.3, right = 0.87, top = 0.92)
            plt.savefig(save_path + 'spatial_periodogram_' + str(int(cluster_id)) +'.png', dpi=300)
            plt.close()
            periodograms.append(powers.T)

    nrows = int(np.ceil(np.sqrt(len(spike_data))))
    ncols = nrows; i=0; j=0
    fig, ax = plt.subplots(ncols=ncols, nrows=nrows, figsize=(20, 20), squeeze=False)
    for periodogram in periodograms:
        vmin, vmax = get_vmin_vmax(periodogram)
        ax[j, i].pcolormesh(X, Y, periodogram, cmap=cmap, shading="auto", vmin=vmin, vmax=vmax)
        i+=1
        if i==ncols:
            i=0; j+=1
    for j in range(nrows):
        for i in range(ncols):
            ax[j, i].spines['top'].set_visible(False) 
            ax[j, i].spines['right'].set_visible(False)
            ax[j, i].spines['bottom'].set_visible(False)
            ax[j, i].spines['left'].set_visible(False)
            ax[j, i].set_xticks([])
            ax[j, i].set_yticks([])
            ax[j, i].xaxis.set_tick_params(labelbottom=False)
            ax[j, i].yaxis.set_tick_params(labelleft=False)
    plt.subplots_adjust(hspace=.1, wspace=.1, bottom=None, left=None, right=None, top=None)
    plt.savefig(save_path + '/all_spatial_periodograms.png', dpi=400)
    plt.close()
  

def plot_firing_rate_maps_per_trial(spike_data, processed_position_data, output_path, track_length):
    save_path = output_path + 'Figures/rate_maps_by_trial'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    rate_maps = []
    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        firing_times_cluster = spike_data["firing_times"].iloc[cluster_index]

        if len(firing_times_cluster)>1:
            cluster_firing_maps = np.array(spike_data['fr_binned_in_space_smoothed'].iloc[cluster_index])
            cluster_firing_maps[np.isnan(cluster_firing_maps)] = 0
            cluster_firing_maps[np.isinf(cluster_firing_maps)] = 0
            percentile_99th_display = np.nanpercentile(cluster_firing_maps, 95);
            cluster_firing_maps = min_max_normalize(cluster_firing_maps)
            percentile_99th = np.nanpercentile(cluster_firing_maps, 95); cluster_firing_maps = np.clip(cluster_firing_maps, a_min=0, a_max=percentile_99th)
            vmin, vmax = get_vmin_vmax(cluster_firing_maps)
            rate_maps.append(cluster_firing_maps)

            spikes_on_track = plt.figure()
            spikes_on_track.set_size_inches(5, 5, forward=True)
            ax = spikes_on_track.add_subplot(1, 1, 1)
            locations = np.arange(0, len(cluster_firing_maps[0]))
            ordered = np.arange(0, len(processed_position_data), 1)
            X, Y = np.meshgrid(locations, ordered)
            cmap = plt.cm.get_cmap("viridis")
            ax.pcolormesh(X, Y, cluster_firing_maps, cmap=cmap, shading="auto", vmin=vmin, vmax=vmax)
            plt.title(str(np.round(percentile_99th_display, decimals=1))+" Hz", fontsize=20)
            plt.ylabel('Trial Number', fontsize=20, labelpad = 20)
            plt.xlabel('Location (cm)', fontsize=20, labelpad = 20)
            plt.xlim(0, track_length)
            ax.tick_params(axis='both', which='both', labelsize=20)
            ax.set_xlim([0, track_length])
            ax.set_ylim([0, len(processed_position_data)-1])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            tick_spacing = 100
            plt.locator_params(axis='y', nbins=3)
            ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
            ax.yaxis.set_ticks_position('left')
            ax.xaxis.set_ticks_position('bottom')
            spikes_on_track.tight_layout(pad=2.0)
            plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.2, right = 0.87, top = 0.92)
            #cbar = spikes_on_track.colorbar(c, ax=ax, fraction=0.046, pad=0.04)
            #cbar.set_label('Firing Rate (Hz)', rotation=270, fontsize=20)
            #cbar.set_ticks([0,np.max(cluster_firing_maps)])
            #cbar.set_ticklabels(["0", "Max"])
            #cbar.ax.tick_params(labelsize=20)
            plt.savefig(save_path + '/firing_rate_map_trials_' + spike_data.session_id.iloc[cluster_index] + '_' + str(int(cluster_id)) + '.png', dpi=300)
            plt.close()

    nrows = int(np.ceil(np.sqrt(len(spike_data))))
    ncols = nrows; i=0; j=0;
    fig, ax = plt.subplots(ncols=ncols, nrows=nrows, figsize=(20, 20), squeeze=False)
    for rate_map in rate_maps:
        vmin, vmax = get_vmin_vmax(rate_map)
        ax[j, i].pcolormesh(X, Y, rate_map, cmap=cmap, shading="auto", vmin=vmin, vmax=vmax)
        i+=1
        if i==ncols:
            i=0; j+=1
    for j in range(nrows):
        for i in range(ncols):
            ax[j, i].spines['top'].set_visible(False)
            ax[j, i].spines['right'].set_visible(False)
            ax[j, i].spines['bottom'].set_visible(False)
            ax[j, i].spines['left'].set_visible(False)
            ax[j, i].set_xticks([])
            ax[j, i].set_yticks([])
            ax[j, i].xaxis.set_tick_params(labelbottom=False)
            ax[j, i].yaxis.set_tick_params(labelleft=False)
    plt.subplots_adjust(hspace=.1, wspace=.1, bottom=None, left=None, right=None, top=None)
    plt.savefig(save_path + '/all_firing_rates.png', dpi=400)
    plt.close()
   
def plot_firing_rate_maps_per_trial_2(cluster_spike_data, track_length, output_path=None, ax=None):
    if output_path is not None:
        save_path = output_path + 'Figures/rate_maps_by_trial'
        if os.path.exists(save_path) is False:
            os.makedirs(save_path)
   
    firing_times_cluster = cluster_spike_data["firing_times"].iloc[0]
    if len(firing_times_cluster)>1:
        cluster_firing_maps = np.array(cluster_spike_data['fr_binned_in_space_smoothed'].iloc[0])
        cluster_firing_maps[np.isnan(cluster_firing_maps)] = 0
        cluster_firing_maps[np.isinf(cluster_firing_maps)] = 0
        percentile_99th_display = np.nanpercentile(cluster_firing_maps, 95)
        cluster_firing_maps = min_max_normalize(cluster_firing_maps)
        percentile_99th = np.nanpercentile(cluster_firing_maps, 95)
        cluster_firing_maps = np.clip(cluster_firing_maps, a_min=0, a_max=percentile_99th)
        vmin, vmax = get_vmin_vmax(cluster_firing_maps)

        locations = np.arange(0, len(cluster_firing_maps[0]))
        ordered = np.arange(0, len(cluster_firing_maps), 1)
        X, Y = np.meshgrid(locations, ordered)
        cmap = plt.cm.get_cmap("viridis")
        
        if ax is None:
            fig = plt.figure()
            fig.set_size_inches(5, 5, forward=True)
            ax = fig.add_subplot(1, 1, 1)

        ax.pcolormesh(X, Y, cluster_firing_maps, cmap=cmap, shading="auto", vmin=vmin, vmax=vmax)
        ax.set_title(str(np.round(percentile_99th_display, decimals=1))+" Hz", fontsize=20)
        ax.set_ylabel('Trial Number', fontsize=20, labelpad = 20)
        ax.set_xlabel('Location (cm)', fontsize=20, labelpad = 20)
        ax.set_xlim([0, track_length])
        ax.set_ylim([0, len(cluster_firing_maps)-1])
        ax.tick_params(axis='both', which='both', labelsize=20)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        tick_spacing = 100
        ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        #cbar = spikes_on_track.colorbar(c, ax=ax, fraction=0.046, pad=0.04)
        #cbar.set_label('Firing Rate (Hz)', rotation=270, fontsize=20)
        #cbar.set_ticks([0,np.max(cluster_firing_maps)])
        #cbar.set_ticklabels(["0", "Max"])
        #cbar.ax.tick_params(labelsize=20)
        if output_path is not None:
            plt.savefig(save_path + '/firing_rate_map_trials_' + 
                        spike_data.session_id.iloc[cluster_index] + '_' + 
                        str(int(cluster_id)) + '.png', dpi=300)

def get_vmin_vmax(cluster_firing_maps, bin_cm=8):
    cluster_firing_maps_reduced = []
    for i in range(len(cluster_firing_maps)):
        cluster_firing_maps_reduced.append(block_reduce(cluster_firing_maps[i], bin_cm, func=np.mean))
    cluster_firing_maps_reduced = np.array(cluster_firing_maps_reduced)
    vmin= 0
    vmax= np.max(cluster_firing_maps_reduced)
    return vmin, vmax 

def plot_combined_behaviour(output_path):
    fig_paths = []
    fig_paths.append(output_path + 'Figures/behaviour/stop_histogram.png')
    fig_paths.append(output_path + 'Figures/behaviour/speed_histogram.png')
    fig_paths.append(output_path + 'Figures/behaviour/speed_per_100ms_hist.png')
    fig_paths.append(output_path + 'Figures/behaviour/x_position_cm_hist.png')

    fig_paths.append(output_path + 'Figures/behaviour/stop_raster.png')
    fig_paths.append(output_path + 'Figures/behaviour/speed_heat_map.png')
    fig_paths.append(output_path + 'Figures/behaviour/trial_number_hist.png')
    fig_paths.append(output_path + 'Figures/behaviour/trial_type_hist.png')
 
    fig_paths.append(output_path + 'Figures/behaviour/lick_hist.png')
    fig_paths.append(output_path + 'Figures/behaviour/lick_raster.png')
    fig_paths.append(output_path + 'Figures/behaviour/pupil_heatmap.png')
    fig_paths.append(output_path + 'Figures/behaviour/pupil_hist.png')

    fig_paths.append(output_path + 'Figures/Sync_test/pulses_after_processing_sync_pulses.png')
    fig_paths.append(output_path + 'Figures/behaviour/hist_barplot.png')
    fig_paths.append(output_path + 'Figures/behaviour/middle_frame.png')
    fig_paths.append(output_path + 'Figures/behaviour/trial_number.png') 
  

    fig, axs = plt.subplots(4, 4, figsize=(10, 9)) 
    for i, fig_path in enumerate(fig_paths):
        row = i // 4
        col = i % 4
        axs[row, col].axis('off')
        if os.path.exists(fig_path):
            img = mpimg.imread(fig_path)
            axs[row, col].imshow(img)
    plt.tight_layout()
    plt.savefig(output_path + 'Figures/behaviour/combined.png', dpi=1000)
    plt.close() 
    
def plot_behaviour(position_data, processed_position_data, output_path, track_length):
    plot_variables(position_data, output_path)
    plot_stops_on_track(processed_position_data, output_path, track_length=track_length)
    plot_stop_histogram(processed_position_data, output_path, track_length=track_length)
    plot_speed_histogram(processed_position_data, output_path, track_length=track_length)
    plot_speed_heat_map(processed_position_data, output_path, track_length=track_length)
     
    #plot_eye_trajectory(position_data, processed_position_data, output_path, track_length=track_length)
    #plot_eye(processed_position_data, output_path, track_length=track_length) 
    plot_combined_behaviour(output_path) 
 
def plot_track_firing(spike_data, processed_position_data, output_path, track_length):  
    #plot_spatial_periodogram_per_trial(spike_data, processed_position_data, output_path, track_length=track_length)
    plot_firing_rate_maps(spike_data, processed_position_data, output_path, track_length=track_length)
    plot_firing_rate_maps_per_trial(spike_data, processed_position_data, output_path, track_length=track_length)
    plot_spikes_on_track(spike_data, processed_position_data, output_path, track_length=track_length)
 

#  this is here for testing
def main():
    print('-------------------------------------------------------------')
    print("8888888") 
    print('-------------------------------------------------------------')


if __name__ == '__main__':
    main()


