from P2_PostProcess.VirtualReality.behaviour_from_blender import *
from P2_PostProcess.VirtualReality.behaviour_from_ADC_channels import *
from P2_PostProcess.VirtualReality.spatial_data import *
from P2_PostProcess.VirtualReality.spatial_firing import *
from P2_PostProcess.VirtualReality.plotting import *
from P2_PostProcess.VirtualReality.video import *

def process(recording_path, processed_folder_name, **kwargs):
    track_length = get_track_length(recording_path)
    stop_threshold = get_stop_threshold(recording_path)

    # look for position_data
    files = [f for f in Path(recording_path).iterdir()]
    if os.path.exists(recording_path+"/"+processed_folder_name+"/position_data_finalised.csv"):
        position_data = pd.read_csv(recording_path+"/"+processed_folder_name+"/position_data.csv")
    elif np.any(["blender.csv" in f.name and f.is_file() for f in files]):
        position_data = generate_position_data_from_blender_file(recording_path, processed_folder_name)
    elif np.any([".continuous" in f.name and f.is_file() for f in files]):
        position_data = generate_position_data_from_ADC_channels(recording_path, processed_folder_name)
    else:
        print("I couldn't find any source of position data")
        return

    print("I am using position data with an avg sampling rate of ", str(1/np.nanmean(np.diff(position_data["time_seconds"]))), "Hz")
    # add a step for syncing data if necesssary
    # TODO position_data = sync_posi...
    # process video
    #position_data = process_video(recording_path, processed_folder_name, position_data)

    for column in list(position_data):
        if "Unnamed" in column:
            del position_data[column]

    # save position data
    position_data.to_csv(recording_path + "/" + processed_folder_name + "/position_data.csv", index=False)

    # process and plot position data
    processed_position_data = process_position_data(position_data, track_length, stop_threshold)
    processed_position_data.to_pickle(recording_path+"/"+processed_folder_name+"/processed_position_data.pkl")
    plot_behaviour(position_data, processed_position_data, output_path=recording_path+"/"+processed_folder_name, track_length=track_length)

    # process and save spatial spike data
    spike_data_path = recording_path+"/"+processed_folder_name+"/"+settings.sorterName+"/firing.pkl"
    if os.path.exists(spike_data_path):
        spike_data = pd.read_pickle(spike_data_path)
        spike_data = add_location_and_task_variables(spike_data, position_data, processed_position_data, track_length)
        spike_data.to_pickle(spike_data_path)

        #plot
        plot_track_firing(spike_data, processed_position_data, output_path=recording_path+"/"+processed_folder_name, track_length=track_length)
        plot_firing_properties(spike_data, output_path=recording_path+"/"+processed_folder_name)
    else:
        print("I couldn't find spike data at ", spike_data_path)
    return

#  this is here for testing
def main():
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')

if __name__ == '__main__':
    main()
