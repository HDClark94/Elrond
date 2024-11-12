from ..VirtualReality.behaviour_from_blender import *
from ..VirtualReality.behaviour_from_ADC_channels import *
from ..VirtualReality.spatial_firing import *
from ..VirtualReality.video import *
from .plotting import *

def process(recording_path, processed_path, **kwargs):
    track_length = get_track_length(recording_path)
    stop_threshold = get_stop_threshold(recording_path)
    # process and save spatial spike data
    if "sorterName" in kwargs.keys():   
        sorterName = kwargs["sorterName"]
    else:
        sorterName = settings.sorterName
    # look for position_data
    files = [f for f in Path(recording_path).iterdir()]
    if np.any(["blender.csv" in f.name and f.is_file() for f in files]):
        position_data = generate_position_data_from_blender_file(recording_path, processed_path)
    elif np.any([".continuous" in f.name and f.is_file() for f in files]):
        position_data = generate_position_data_from_ADC_channels(recording_path, processed_path)
    else: 
        print("I couldn't find any source of position data")
        return
    print("I am using position data with an avg sampling rate of ", str(1/np.nanmean(np.diff(position_data["time_seconds"]))), "Hz")
    # process video
    #position_data = process_video(recording_path, processed_path, position_data, 
    #                              pupil_model_path=kwargs["deeplabcut_vr_pupil_model_path"],
    #                              licks_model_path=kwargs["deeplabcut_vr_licks_model_path"])  
    position_data_path = processed_path + "position_data.csv"
    processed_position_data_path = processed_path + "processed_position_data.pkl"
    spike_data_path = processed_path + sorterName + "/spikes.pkl"
    # save position data
    position_data.to_csv(position_data_path, index=False)
    position_data["syncLED"] = position_data["sync_pulse"]
    # process and plot position data
    processed_position_data = process_position_data(position_data, track_length, stop_threshold)
    processed_position_data.to_pickle(processed_position_data_path)
    plot_behaviour(position_data, processed_position_data, output_path=processed_path, track_length=track_length)
    # process and save spatial spike data
    if os.path.exists(spike_data_path):
        spike_data = pd.read_pickle(spike_data_path)
        position_data = synchronise_position_data_via_ADC_ttl_pulses(position_data, processed_path, recording_path)
        spike_data = add_location_and_task_variables(spike_data, position_data, processed_position_data, track_length)
        position_data.to_csv(position_data_path, index=False)
        spike_data.to_pickle(spike_data_path)    
        plot_track_firing(spike_data, processed_position_data, output_path=processed_path + sorterName+"/", track_length=track_length)
    else: 
        print("I couldn't find spike data at ", spike_data_path)
    return
#  this is here for testing 
def main():
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')
if __name__ == '__main__':
    main()