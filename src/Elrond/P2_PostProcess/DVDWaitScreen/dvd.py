from Elrond.P2_PostProcess.Shared.time_sync import *
from Elrond.P2_PostProcess.DVDWaitScreen.spatial_data import *
from Elrond.P2_PostProcess.OpenField.spatial_firing import *
from Elrond.P2_PostProcess.OpenField.plotting import *
from Elrond.P2_PostProcess.DVDWaitScreen.behaviour_from_blender import *


def plot_behaviour(position_data, output_path):
    save_path = output_path + '/Figures/firing_scatters'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    raw_trajectory = plt.figure()
    raw_trajectory.set_size_inches(5, 5, forward=True)
    ax = raw_trajectory.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
    ax.plot(position_data['position_x'], position_data['position_y'], color='black', linewidth=2, zorder=1, alpha=0.7)
    plt.title('Trajectory', y=1.08, fontsize=24)
    plt.savefig(save_path + '/trajectory.png',dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()
    return
 
def process(recording_path, processed_path, **kwargs):
    files = [f for f in Path(recording_path).iterdir()]
    if np.any(["blender.csv" in f.name and f.is_file() for f in files]):
        position_data = generate_position_data_from_blender_file(recording_path, processed_path)
        print("found blender file")
        
    # process and save position data
    position_data = process_position_data(position_data, **kwargs)
    plot_behaviour(position_data, output_path=processed_path)

    position_data = synchronise_position_data_via_ADC_ttl_pulses(position_data, processed_path, recording_path)
    position_heat_map = get_position_heatmap(position_data)

    # save position data
    position_data.to_csv(processed_path + "position_data.csv")
    print("saved syned position_data") 

    # process and save spatial spike data
    if "sorterName" in kwargs.keys():
        sorterName = kwargs["sorterName"]
    else:
        sorterName = settings.sorterName

    spike_data_path = processed_path + sorterName+"/spikes.pkl"
    if os.path.exists(spike_data_path):

        output_path = processed_path + sorterName + "/"
        spike_data = pd.read_pickle(spike_data_path)

        spike_data = add_spatial_variables(spike_data, position_data)
        plot_firing_rate_maps(spike_data, output_path)

        spike_data = add_scores(spike_data, position_data, position_heat_map)
        spike_data.to_pickle(spike_data_path)

        plot_rate_map_autocorrelogram(spike_data, output_path)
        plot_spikes_on_trajectory(spike_data, position_data, output_path)
        plot_coverage(position_heat_map, output_path)
        plot_polar_head_direction_histogram(spike_data, position_data, output_path)
        plot_firing_rate_vs_speed(spike_data, position_data, output_path)
        make_combined_figure(spike_data, output_path)
    else:
        print("I couldn't find spike data at ", spike_data_path)
    return



def main():
    position_data = pd.read_csv("/mnt/datastore/Harry/Cohort12_august2024/derivatives/M22/D50/dvd_waitscreen/M22_D50_2024-11-18_15-25-24_DVD/processed/position_data.csv")
    spike_data = pd.read_pickle("/mnt/datastore/Harry/Cohort12_august2024/derivatives/M22/D50/dvd_waitscreen/M22_D50_2024-11-18_15-25-24_DVD/processed/kilosort4/spikes.pkl")
    spike_data = add_spatial_variables(spike_data, position_data)
    print("hello")


if __name__ == '__main__':
    main()