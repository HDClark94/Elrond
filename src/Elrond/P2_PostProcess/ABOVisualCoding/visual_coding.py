from .behaviour_from_blender import *
from ..VirtualReality.spatial_firing import *
from ..VirtualReality.video import *

from scipy import stats
#
import matplotlib.image as mpimg

def plot_ranked_image_peristimulus_by_shank(spike_data, position_data, output_path, top_n=10):
    save_path = output_path + '/Figures/ranked_image_peristimulus'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    # Create a figure and a set of subplots with 3 rows and 6 columns
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.axvline(x=0.25, linewidth=1, color="grey")
    ax.axvline(x=0.50, linewidth=1, color="grey")
    shank_colors = ["blue", "red", "yellow", "green"]

    recording_length = max(position_data["time_seconds"])
    for shuffle, shuffle_linestyle in zip(["", "_shuffled"], ["solid", "dashed"]):
        for shank_idx, shank_id in enumerate(np.unique(spike_data.shank_id)):
            print(shank_id)
            shank_spike_data = spike_data[spike_data.shank_id == shank_id]

            shank_hists = []
            for cluster_index, cluster_id in enumerate(shank_spike_data.cluster_id):
                cluster_spike_data = shank_spike_data[shank_spike_data["cluster_id"] == cluster_id]
                firing_times_cluster = np.array(cluster_spike_data["firing_times"].iloc[0])/settings.sampling_rate
                if shuffle == "_shuffled":
                    random_firing_additions =  np.random.uniform(low=20, high=recording_length-20)
                    firing_times_cluster += random_firing_additions
                    firing_times_cluster[firing_times_cluster >= recording_length] -= recording_length  # wrap around

                if len(firing_times_cluster) > 1:
                    ranked_images = cluster_spike_data["image_ranks"].iloc[0]
                    top_10_ranked_images = ranked_images[::-1][:top_n]

                    for ranked_image in top_10_ranked_images:
                        id_trial_numbers = np.unique(position_data[position_data["image_ID"] == ranked_image]["trial_number"])

                        valid_times_all_trials = []
                        for tn_idx, tn in enumerate(id_trial_numbers):
                            t_start = position_data[position_data["trial_number"] == tn]["time_seconds"].iloc[0]-0.25
                            t_end = t_start+1.75
                            valid_times = firing_times_cluster[(firing_times_cluster > t_start) &
                                                               (firing_times_cluster < t_end)]
                            valid_times = valid_times-t_start # offset to trial tn-1 start
                            valid_times_all_trials.extend(valid_times.tolist())
                        valid_times_all_trials = np.array(valid_times_all_trials)
                        time_bins = np.arange(0, 2, settings.time_bin_size)  # 100ms time bins

                        hist, bin_edges = np.histogram(valid_times_all_trials, time_bins)
                        bin_centres = 0.5 * (bin_edges[1:] + bin_edges[:-1])
                        shank_hists.append(hist.tolist())
            shank_hists = np.array(shank_hists)
            shank_hists = stats.zscore(shank_hists, axis=1, nan_policy="omit")

            ax.plot(bin_centres, np.nanmean(shank_hists, axis=0), color=shank_colors[shank_idx],
                    label="shank_id="+str(shank_id)+str(shuffle), linestyle=shuffle_linestyle)
            ax.fill_between(bin_centres,
                            np.nanmean(shank_hists, axis=0)-stats.sem(shank_hists, axis=0, nan_policy="omit"),
                            np.nanmean(shank_hists, axis=0)+stats.sem(shank_hists, axis=0, nan_policy="omit"),
                            color=shank_colors[shank_idx], alpha=0.2)
    ax.set_xlim(0, 1.5)
    ax.set_ylim(bottom=-0.5)

    ax.set_title("top " + str(top_n) + " image responses across shanks")
    ax.set_ylabel("z scored fr")
    ax.set_xlabel("time (s)")
    ax.legend()

    # Adjust layout to prevent overlapping
    plt.tight_layout()
    plt.savefig(save_path + '_perstimulus_shank_comparison_top_' + str(top_n) + '.png', dpi=300)
    plt.close()


def plot_ranked_image_peristimulus_plots(spike_data, position_data, output_path):
    save_path = output_path + '/Figures/ranked_image_peristimulus'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        cluster_spike_data = spike_data[spike_data["cluster_id"] == cluster_id]
        shank_id = cluster_spike_data["shank_id"].iloc[0]


        firing_times_cluster = np.array(cluster_spike_data["firing_times"].iloc[0])/settings.sampling_rate
        firing_trial_numbers_cluster = np.array(cluster_spike_data["trial_number"].iloc[0])
        image_IDs_cluster = np.array(cluster_spike_data["image_ID"].iloc[0])

        if len(firing_times_cluster) > 1:
            fr_binned_in_time = cluster_spike_data["fr_time_binned"].iloc[0]
            ranked_images = cluster_spike_data["image_ranks"].iloc[0]

            # Create a figure and a set of subplots with 3 rows and 6 columns
            fig, axs = plt.subplots(3, 6, figsize=(18, 9))

            # Set the main title of the figure
            fig.suptitle('Figure with Images and Line Plots', fontsize=16)

            # Display images in the first row
            for ax, ranked_image in zip(axs[0], ranked_images[::-1][:6]):
                img = mpimg.imread(settings.natural_scenes_image_folder_path+"/image_" + str(-1 * ranked_image) + ".jpg")
                ax.imshow(img, cmap="gray")
                ax.axis('off')  # Hide the axes for images

            # spike raster over 50 trials
            for ax, ranked_image in zip(axs[1], ranked_images[::-1][:6]):
                ax.axvline(x=0.25, linewidth=1, color="grey")
                ax.axvline(x=0.50, linewidth=1, color="grey")
                ax.set_xlim(0, 1.5)
                ax.set_ylim(0, 50)
                id_trial_numbers = np.unique(position_data[position_data["image_ID"] == ranked_image]["trial_number"])
                n_spikes_collected = []
                for tn_idx, tn in enumerate(id_trial_numbers):
                    t_start = position_data[position_data["trial_number"] == tn]["time_seconds"].iloc[0]-0.25
                    t_end = t_start+1.5

                    valid_times = firing_times_cluster[(firing_times_cluster > t_start) &
                                                       (firing_times_cluster < t_end)]
                    if len(valid_times)>0:
                        valid_times = valid_times - t_start  # offset to trial tn-1 start
                        ax.plot(valid_times, np.ones(len(valid_times))*tn_idx, "|", markersize=4, color="black")
                        n_spikes_collected.extend(valid_times)
                n_spikes_collected = np.array(n_spikes_collected)

            # spike histogram over
            for ax, ranked_image in zip(axs[2], ranked_images[::-1][:6]):
                ax.axvline(x=0.25, linewidth=1, color="grey")
                ax.axvline(x=0.50, linewidth=1, color="grey")
                ax.set_xlim(0,1.5)

                id_trial_numbers = np.unique(position_data[position_data["image_ID"] == ranked_image]["trial_number"])
                valid_times_all_trials = []
                for tn_idx, tn in enumerate(id_trial_numbers):
                    t_start = position_data[position_data["trial_number"] == tn]["time_seconds"].iloc[0]-0.25
                    t_end = t_start+1.5

                    valid_times = firing_times_cluster[(firing_times_cluster > t_start) &
                                                       (firing_times_cluster < t_end)]
                    valid_times = valid_times-t_start # offset to trial tn-1 start
                    valid_times_all_trials.extend(valid_times.tolist())
                valid_times_all_trials = np.array(valid_times_all_trials)
                time_bins = np.arange(0, 1.75, settings.time_bin_size)  # 100ms time bins

                hist, bin_edges = np.histogram(valid_times_all_trials, time_bins)
                bin_centres = 0.5 * (bin_edges[1:] + bin_edges[:-1])
                ax.bar(bin_centres, hist, width=settings.time_bin_size, color="black")

            # Adjust layout to prevent overlapping
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.tight_layout()
            plt.savefig(save_path + '/' + spike_data.session_id.iloc[cluster_index] +
                        '_perstimulus_shank' + str(shank_id) + '_c' + str(cluster_id) + '.png', dpi=300)
            plt.close()
    return


def plot_firing(spike_data, position_data, output_path):
    print('I am plotting firing rate maps...')
    save_path = output_path + '/Figures/spike_rate'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        cluster_spike_data = spike_data[spike_data["cluster_id"] == cluster_id]
        firing_times_cluster = np.array(cluster_spike_data["firing_times"].iloc[0])

        if len(firing_times_cluster) > 1:
            fr_binned_in_time = cluster_spike_data["fr_time_binned"].iloc[0]

            spikes_on_track = plt.figure()
            spikes_on_track.set_size_inches(5, 5, forward=True)
            ax = spikes_on_track.add_subplot(1, 1, 1)
            y_max = 0

            trial_numbers = np.unique(position_data["trial_number"])
            all_fr_binned_in_time = list_of_list_to_1d_numpy_array_from_indices(fr_binned_in_time, trial_numbers - 1)
            avg_rate = np.nanmean(all_fr_binned_in_time)
            ax.axhline(avg_rate, color="grey", linestyle="dashed", linewidth=2)

            for id in np.unique(position_data["image_ID"]):
                id_trial_numbers = np.unique(position_data[position_data["image_ID"] == id]["trial_number"])
                id_fr_binned_in_time = list_of_list_to_1d_numpy_array_from_indices(fr_binned_in_time, id_trial_numbers - 1)

                ax.errorbar(id, np.nanmean(id_fr_binned_in_time),
                            yerr=stats.sem(id_fr_binned_in_time, axis=0, nan_policy="omit"), color="black")

                y_max = max([y_max, np.nanmean(id_fr_binned_in_time)])
                y_max = np.ceil(y_max)

            plt.ylabel('Firing Rate (Hz)', fontsize=20, labelpad=20)
            plt.xlabel('ID', fontsize=20, labelpad=20)
            ax.yaxis.set_ticks_position('left')
            ax.xaxis.set_ticks_position('bottom')
            ax.set_yticks([0, np.round(ax.get_ylim()[1], 2)])
            plt.locator_params(axis='y', nbins=4)
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.tight_layout()
            plt.savefig(save_path + '/' + spike_data.session_id.iloc[cluster_index] + '_rate_map_Cluster_' + str(
                cluster_id) + '.png', dpi=300)
            plt.close()
    return

def plot_firing2(spike_data, position_data, output_path):
    print('I am plotting firing rate maps...')
    save_path = output_path + '/Figures/number_of_spikes'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        cluster_spike_data = spike_data[spike_data["cluster_id"] == cluster_id]
        firing_times_cluster = np.array(cluster_spike_data["firing_times"].iloc[0])
        image_IDs = np.array(cluster_spike_data["image_ID"].iloc[0])
        trial_numbers = np.array(cluster_spike_data["trial_number"].iloc[0])

        if len(firing_times_cluster) > 1:
            fr_binned_in_time = cluster_spike_data["fr_time_binned"].iloc[0]

            spikes_on_track = plt.figure()
            spikes_on_track.set_size_inches(5, 5, forward=True)
            ax = spikes_on_track.add_subplot(1, 1, 1)
            y_max = 0

            trial_numbers = np.unique(position_data["trial_number"])
            all_fr_binned_in_time = list_of_list_to_1d_numpy_array_from_indices(fr_binned_in_time, trial_numbers - 1)
            avg_rate = np.nanmean(all_fr_binned_in_time)
            #ax.axhline(avg_rate, color="grey", linestyle="dashed", linewidth=2)

            n_spikes_all = []
            for id in np.unique(position_data["image_ID"]):
                n_spikes = len(firing_times_cluster[image_IDs == id])
                n_spikes_all.append(n_spikes)

                id_trial_numbers = np.unique(position_data[position_data["image_ID"] == id]["trial_number"])
                id_fr_binned_in_time = list_of_list_to_1d_numpy_array_from_indices(fr_binned_in_time, id_trial_numbers - 1)
                ax.plot(id, n_spikes, marker="|", color="black")
            n_spikes_all = np.array(n_spikes_all)

            plt.ylabel('Firing Rate (Hz)', fontsize=20, labelpad=20)
            plt.xlabel('ID', fontsize=20, labelpad=20)
            ax.yaxis.set_ticks_position('left')
            ax.xaxis.set_ticks_position('bottom')
            ax.set_yticks([0, np.round(ax.get_ylim()[1], 2)])
            plt.locator_params(axis='y', nbins=4)
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.tight_layout()
            plt.savefig(save_path + '/' + spike_data.session_id.iloc[cluster_index] + '_rate_map_Cluster_' + str(cluster_id) + '.png', dpi=300)
            plt.close()
    return

def add_kinematics(spike_data, position_data):
    position_sampling_rate = float(1/np.mean(np.diff(position_data["time_seconds"])))

    trial_number = []
    imageIDs = []
    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        cluster_firing_indices = np.asarray(spike_data[spike_data.cluster_id == cluster_id].firing_times)[0]
        cluster_firing_seconds = cluster_firing_indices/settings.sampling_rate
        cluster_firing_position_data_indices = np.array(np.round(cluster_firing_seconds*position_sampling_rate), dtype=np.int64)
        trial_number.append(position_data["trial_number"][cluster_firing_position_data_indices].to_list())
        imageIDs.append(position_data["image_ID"][cluster_firing_position_data_indices].to_list())
    spike_data["trial_number"] = trial_number
    spike_data["image_ID"] = imageIDs
    return spike_data


def add_location_and_task_variables(spike_data, position_data):
    spike_data = add_kinematics(spike_data, position_data)
    spike_data = bin_fr_in_time(spike_data, position_data, smoothen=True)
    spike_data = bin_fr_in_time(spike_data, position_data, smoothen=False)
    return spike_data


def bin_fr_in_time(spike_data, position_data, smoothen=True):
    if smoothen:
        suffix="_smoothed"
    else:
        suffix=""

    gauss_kernel = Gaussian1DKernel(settings.guassian_std_for_smoothing_in_time_seconds/settings.time_bin_size)
    n_trials = max(position_data["trial_number"])

    # make an empty list of list for all firing rates binned in time for each cluster
    fr_binned_in_time = [[] for x in range(len(spike_data))]

    # extract spatial variables from position
    times = np.array(position_data['time_seconds'], dtype="float64")
    trial_numbers_raw = np.array(position_data['trial_number'], dtype=np.int64)

    # calculate the average fr in each 100ms time bin
    time_bins = np.arange(min(times), max(times), settings.time_bin_size) # 100ms time bins
    tn_time_bin_means = (np.histogram(times, time_bins, weights = trial_numbers_raw)[0] / np.histogram(times, time_bins)[0]).astype(np.int64)

    for i, cluster_id in enumerate(spike_data.cluster_id):
        if len(time_bins)>1:
            spike_times = np.array(spike_data[spike_data["cluster_id"] == cluster_id]["firing_times"].iloc[0])
            spike_times = spike_times/settings.sampling_rate # convert to seconds

            #===========================
            shuffle=True
            if shuffle == True:
                recording_length = max(position_data["time_seconds"])
                random_firing_additions = np.random.uniform(low=20, high=recording_length - 20)
                spike_times += random_firing_additions
                spike_times[spike_times >= recording_length] -= recording_length  # wrap around
            #===========================

            # count the spikes in each time bin and normalise to seconds
            fr_time_bin_means, bin_edges = np.histogram(spike_times, time_bins)
            fr_time_bin_means = fr_time_bin_means/settings.time_bin_size

            # and smooth
            if smoothen:
                fr_time_bin_means = convolve(fr_time_bin_means, gauss_kernel)

            # fill in firing rate array by trial
            fr_binned_in_time_cluster = []
            for trial_number in range(1, n_trials+1):
                fr_binned_in_time_cluster.append(fr_time_bin_means[tn_time_bin_means == trial_number].tolist())
            fr_binned_in_time[i] = fr_binned_in_time_cluster
        else:
            fr_binned_in_time[i] = []

    spike_data["fr_time_binned"+suffix] = fr_binned_in_time
    return spike_data

def rank_image_firing(spike_data, position_data):
    image_ranks = []
    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        cluster_spike_data = spike_data[spike_data["cluster_id"] == cluster_id]
        firing_times_cluster = np.array(cluster_spike_data["firing_times"].iloc[0])

        if len(firing_times_cluster) > 1:
            fr_binned_in_time = cluster_spike_data["fr_time_binned"].iloc[0]

            peak_firing_by_ID = []
            for id in np.unique(position_data["image_ID"]):
                id_trial_numbers = np.unique(position_data[position_data["image_ID"] == id]["trial_number"])

                avg_rates = []
                # we want to look at the firing rates before, during and after any given image stimuli
                for tn in [-1,0,1,2,3,4]:
                    tns= id_trial_numbers-tn
                    tns = np.intersect1d(tns, np.unique(position_data["trial_number"])) # filter out non trial numbers

                    # calculate the time binned firing rates for a given set of trials
                    id_fr_binned_in_time = list_of_list_to_1d_numpy_array_from_indices(fr_binned_in_time, tns-1)

                    # calculate the firing rate mean across those trials
                    avg_rates.append(np.nanmean(id_fr_binned_in_time))
                avg_rates = np.array(avg_rates)

                # calculate the peak firing
                peak_firing = np.nanmax(avg_rates)

                peak_firing_by_ID.append(peak_firing)
            peak_firing_by_ID = np.array(peak_firing_by_ID)
            ranked_IDs = np.unique(position_data["image_ID"])[np.argsort(peak_firing_by_ID)]
            ranked_IDs = ranked_IDs.tolist()
        else:
            ranked_IDs = []

        image_ranks.append(ranked_IDs)
    spike_data["image_ranks"] = image_ranks
    return spike_data


def process(recording_path, processed_path, **kwargs):
    # process and save spatial spike data
    if "sorterName" in kwargs.keys():
        sorterName = kwargs["sorterName"]
    else:
        sorterName = settings.sorterName

    # look for position_data
    files = [f for f in Path(recording_path).iterdir()]
    if np.any(["blender.csv" in f.name and f.is_file() for f in files]):
        position_data = generate_position_data_from_blender_file(recording_path, processed_path)
    else:
        print("I couldn't find any source of position data")

    print("I am using position data with an avg sampling rate of ", str(1/np.nanmean(np.diff(position_data["time_seconds"]))), "Hz")

    # process video
    #position_data = process_video(recording_path, processed_folder_name, position_data)

    position_data_path = processed_path + "position_data.csv"
    spike_data_path = processed_path + sorterName + "/spikes.pkl"

    # save position data
    position_data.to_csv(position_data_path, index=False)
    position_data["syncLED"] = position_data["sync_pulse"]

    # process and save spatial spike data
    if os.path.exists(spike_data_path):
        spike_data = pd.read_pickle(spike_data_path)
        position_data = synchronise_position_data_via_ADC_ttl_pulses(position_data, processed_path, recording_path)
        spike_data = add_location_and_task_variables(spike_data, position_data)
        position_data.to_csv(position_data_path, index=False)

        spike_data = rank_image_firing(spike_data, position_data)
        spike_data.to_pickle(spike_data_path)

        plot_ranked_image_peristimulus_plots(spike_data, position_data, output_path=recording_path+"/"+processed_path)
        plot_ranked_image_peristimulus_by_shank(spike_data, position_data, output_path=recording_path + "/" + processed_folder_name, top_n=n)
        plot_firing(spike_data, position_data, output_path=recording_path+"/"+processed_folder_name)
        #plot_firing2(spike_data, position_data, output_path=recording_path+"/"+processed_folder_name)
    else:
        print("I couldn't find spike data at ", spike_data_path)
    return

#  this is here for testing
def main():
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')

if __name__ == '__main__':
    main()
