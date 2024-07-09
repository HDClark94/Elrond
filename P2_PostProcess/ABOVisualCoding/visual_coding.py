from P2_PostProcess.ABOVisualCoding.behaviour_from_blender import *

#from P2_PostProcess.VirtualReality.spatial_firing import *
#from P2_PostProcess.VirtualReality.plotting import *
#from P2_PostProcess.VirtualReality.video import *

def process(recording_path, processed_folder_name, **kwargs):
    # process and save spatial spike data
    if "sorterName" in kwargs.keys():
        sorterName = kwargs["sorterName"]
    else:
        sorterName = settings.sorterName

    # look for position_data
    files = [f for f in Path(recording_path).iterdir()]
    if np.any(["blender.csv" in f.name and f.is_file() for f in files]):
        position_data = generate_position_data_from_blender_file(recording_path, processed_folder_name)
    print("I couldn't find any source of position data")

    print("I am using position data with an avg sampling rate of ", str(1/np.nanmean(np.diff(position_data["time_seconds"]))), "Hz")

    # process video
    #position_data = process_video(recording_path, processed_folder_name, position_data)
    for column in list(position_data):
        if "Unnamed" in column:
            del position_data[column]

    position_data_path = recording_path + "/" + processed_folder_name + "/position_data.csv"
    processed_position_data_path = recording_path + "/" + processed_folder_name + "/processed_position_data.pkl"
    spike_data_path = recording_path + "/" + processed_folder_name + "/" + sorterName + "/spikes.pkl"

    # save position data
    position_data.to_csv(position_data_path, index=False)
    position_data["syncLED"] = position_data["sync_pulse"]

    # process and plot position data
    processed_position_data = process_position_data(position_data, track_length, stop_threshold)
    processed_position_data.to_pickle(processed_position_data_path)
    plot_behaviour(position_data, processed_position_data, output_path=recording_path+"/"+processed_folder_name, track_length=track_length)

    # process and save spatial spike data
    if os.path.exists(spike_data_path) and not ("postprocess_behaviour_only" in kwargs and kwargs["postprocess_behaviour_only"]):
        spike_data = pd.read_pickle(spike_data_path)
        position_data = synchronise_position_data_via_ADC_ttl_pulses(position_data, processed_folder_name, recording_path)
        spike_data = add_location_and_task_variables(spike_data, position_data, processed_position_data, track_length)
        position_data.to_csv(position_data_path, index=False)
        spike_data.to_pickle(spike_data_path)
        plot_track_firing(spike_data, processed_position_data, output_path=recording_path+"/"+processed_folder_name, track_length=track_length)
    else:
        print("I couldn't find spike data at ", spike_data_path)
    return

#  this is here for testing
def main():
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')

if __name__ == '__main__':
    main()
