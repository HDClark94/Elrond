from P2_PostProcess.Shared.time_sync import *
from P2_PostProcess.OpenField.spatial_data import *
from P2_PostProcess.OpenField.spatial_firing import *
from P2_PostProcess.OpenField.plotting import *

def process(recording_path, processed_folder_name, **kwargs):

    # process and save position data
    position_data = process_position_data(recording_path, processed_folder_name, **kwargs)
    position_data = synchronise_position_data_via_ADC_ttl_pulses(position_data, processed_folder_name, recording_path)
    position_heat_map = get_position_heatmap(position_data)
    # save position data
    position_data.to_csv(recording_path + "/" + processed_folder_name + "/position_data.csv")

    # process and save spatial spike data
    if "sorterName" in kwargs.keys():
        sorterName = kwargs["sorterName"]
    else:
        sorterName = settings.sorterName

    spike_data_path = recording_path+"/"+processed_folder_name+"/"+sorterName+"/spikes.pkl"
    if os.path.exists(spike_data_path) and not ("postprocess_behaviour_only" in kwargs and kwargs["postprocess_behaviour_only"]):
        spike_data = pd.read_pickle(spike_data_path)
        spike_data = add_spatial_variables(spike_data, position_data)
        spike_data = add_scores(spike_data, position_data)
        spike_data.to_pickle(spike_data_path)

        # make plots
        output_path = recording_path+"/"+processed_folder_name+"/"+sorterName
        plot_spikes_on_trajectory(spike_data, position_data, output_path)
        plot_coverage(position_heat_map, output_path)
        plot_firing_rate_maps(spike_data, output_path)
        plot_rate_map_autocorrelogram(spike_data, output_path)
        plot_polar_head_direction_histogram(spike_data, position_data, output_path)
        plot_firing_rate_vs_speed(spike_data, position_data, output_path)
        make_combined_figure(spike_data, output_path)
    else:
        print("I couldn't find spike data at ", spike_data_path)
    return