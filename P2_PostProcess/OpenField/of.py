from P2_PostProcess.Shared.time_sync import *
from P2_PostProcess.OpenField.spatial_data import *
from P2_PostProcess.OpenField.spatial_firing import *
from P2_PostProcess.OpenField.plotting import *

def process(recording_path, processed_folder_name, **kwargs):

    # process and save position data
    position_data = process_position_data(recording_path, **kwargs)
    position_data = synchronise_position_data_via_ADC_ttl_pulses(position_data, recording_path)
    position_heat_map = get_position_heatmap(position_data)

    # save position data
    position_data.to_csv(recording_path + "/" + processed_folder_name + "/position_data.csv")

    # process and save spatial spike data
    spike_data_path = recording_path+"/"+processed_folder_name+"/"+settings.sorterName+"/firing.pkl"
    if os.path.exists(spike_data_path):
        spike_data = pd.read_pickle(spike_data_path)
        spike_data = add_spatial_variables(spike_data, position_data)
        spike_data = add_scores(spike_data, position_data)
        spike_data.to_pickle(spike_data_path)

        # make plots
        plot_firing_properties(spike_data, output_path=recording_path + "/" + processed_folder_name)
    else:
        print("I couldn't find spike data at ", spike_data_path)
    return