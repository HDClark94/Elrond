from P2_PostProcess.Shared.time_sync import *
from P2_PostProcess.OpenField.spatial_data import *
from P2_PostProcess.OpenField.spatial_firing import *
from P2_PostProcess.OpenField.plotting import *

def process(recording_path, processed_folder_name, **kwargs):

    # process and save position data
    position_data = process_position_data(recording_path)
    # add a step for syncing data if necesssary
    # TODO position_data = sync_posi...
    position_data = synchronise_position_data_via_ttl_pulses(position_data, recording_path)

    position_data.to_pickle(recording_path+"/"+processed_folder_name+"/position_data.pkl")

    # process and save spatial spike data
    spike_data_path = recording_path+"/"+processed_folder_name+"/"+settings.sorterName+"/firing.pkl"
    if os.path.exists(spike_data_path):
        spike_data = pd.read_pickle(spike_data_path)
        #spike_data = # add spatial variables?
        spike_data.to_pickle(spike_data_path)
    else:
        print("I couldn't find spike data at ", spike_data_path)

    # make plots
    plot_firing_properties(spike_data, output_path=recording_path+"/"+processed_folder_name)
    return