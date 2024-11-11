from .behaviour_from_blender import *
from .behaviour_from_ADC_channels import *
from .spatial_firing import *
from .plotting import *
from .video import *
from Elrond.P3_CurrentAnalysis.basic_lomb_scargle_estimator import lomb_scargle
from Elrond.P3_CurrentAnalysis.ramp_score import calculate_ramp_scores, calculate_ramp_scores_parallel

def process(recording_path, processed_path, dlc_data=None, **kwargs):
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

    if dlc_data is not None:
        # syncrhonise position data and video data
        position_data, video_data = synchronise_position_data_via_column_ttl_pulses(position_data, video_data, processed_path, recording_path)
        # now video data contains a synced_time column which is relative to the start of the time column in position_data
        position_data = add_synced_videodata_to_position_data(position_data, video_data)

    print("I am using position data with an avg sampling rate of ", str(1/np.nanmean(np.diff(position_data["time_seconds"]))), "Hz")
  
    position_data_path = processed_path + "position_data.csv"
    processed_position_data_path = processed_path + "processed_position_data.pkl"
    spike_data_path = processed_path + sorterName + "/spikes.pkl"

    # save position data
    position_data.to_csv(position_data_path, index=False)
    position_data["syncLED"] = position_data["sync_pulse"]

    # get and sync lick data
    lick_train_path = processed_path + "../licks/lick_train.npy"
    if Path(lick_train_path).exists() is False:
        try:
            generate_lick_train(
                lick_video_path=kwargs["lick_video_path"], 
                lick_model_folder=kwargs["lick_model_folder"], 
                lick_output_folder=kwargs["lick_output_folder"]
            )
        except Exception as error:
            print(f"Could not make lick train:\n {error}")

    try:
        bonsai_csv_paths = [os.path.abspath(os.path.join(recording_path, filename)) for filename in os.listdir(recording_path) if filename.endswith("_capture.csv")]
        bonsai_data = read_bonsai_file(bonsai_csv_paths[0])
        lick_train = np.load(lick_train_path)
        lick_data = make_lick_data(lick_train, bonsai_data)
        synced_lick_train = get_synced_lick_train(position_data, lick_data, processed_path, recording_path)
    except:
        synced_lick_train = None

    # process and plot position data
    processed_position_data = process_position_data(position_data, track_length, stop_threshold)
    processed_position_data.to_pickle(processed_position_data_path)
    plot_behaviour(position_data, processed_position_data, output_path=processed_path, track_length=track_length)
    if synced_lick_train is not None:
        plot_licks_on_track(processed_position_data, synced_lick_train, processed_path + "Figures/behaviour/", track_length=track_length)

    # process and save spatial spike data
    if os.path.exists(spike_data_path):
        spike_data = pd.read_pickle(spike_data_path)
        position_data = synchronise_position_data_via_ADC_ttl_pulses(position_data, processed_path, recording_path)
        spike_data = add_location_and_task_variables(spike_data, position_data, processed_position_data, track_length)
        spike_data = lomb_scargle(spike_data, processed_position_data, track_length)
        position_data.to_csv(position_data_path, index=False)
        spike_data.to_pickle(spike_data_path)    
        _ = calculate_ramp_scores_parallel(spike_data, processed_position_data, position_data, track_length,save_path=processed_path+sorterName+"/", save=True)
        plot_track_firing(spike_data, processed_position_data, output_path=processed_path + sorterName+"/", track_length=track_length)
    else: 
        print("I couldn't find spike data at ", spike_data_path)
    return


def generate_lick_train(lick_video_path, lick_model_folder, lick_output_folder):

    import deeplabcut as dlc

    config_path = lick_model_folder + "config.yaml"

    dlc.analyze_videos(config_path, [lick_video_path], save_as_csv=True, destfolder = lick_output_folder)
    dlc.filterpredictions(config_path, [lick_video_path])
    dlc.create_labeled_video(config_path, [lick_video_path])
    dlc.plot_trajectories(config_path, [lick_video_path])

    threshold=15

    files_in_folder = os.listdir(lick_output_folder)

    tongue_location_path = lick_output_folder + [s for s in files_in_folder if ".csv" in s][0]

    tong_pos = pd.read_csv(tongue_location_path)

    x_head, y_head = tong_pos.keys()[1], tong_pos.keys()[2]

    xs = tong_pos[x_head][2:].values.astype('float')
    ys = tong_pos[y_head][2:].values.astype('float')

    lick_train = np.arange(1,len(ys))[np.diff(ys) > threshold]
    np.save(lick_output_folder + "lick_train.npy", lick_train)


def make_lick_data(lick_train, bonsai_data):
    video_frames = len(bonsai_data)

    lick_frames = np.zeros(video_frames, dtype=np.int8)
    for i, frame in enumerate(lick_frames):
        if i in lick_train:
            lick_frames[i] = 1

    lick_data = pd.DataFrame(lick_frames)
    lick_data['syncLED'] = bonsai_data['syncLED']
    lick_data['time_seconds'] = bonsai_data['time_seconds']

    return lick_data

def get_synced_lick_train(position_data, lick_data, processed_path, recording_path):

    _, lick_data = synchronise_position_data_via_column_ttl_pulses(position_data, lick_data, processed_path, recording_path)

    synced_lick_train = []
    for i , row in lick_data.iterrows():
        if row[0] == 1:
            synced_lick_train.append( row['synced_time'])

    return synced_lick_train


#  this is here for testing
def main():
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')

if __name__ == '__main__':
    main()
