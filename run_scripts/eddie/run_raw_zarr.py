from pathlib import Path
import spikeinterface.full as si
from Elrond.Helpers.upload_download import get_chronologized_recording_paths
import sys


mouse = sys.argv[1]
day = sys.argv[2]
project_path = sys.argv[3]

si.set_global_job_kwargs(n_jobs=8)

raw_recordings = get_chronologized_recording_paths(project_path, mouse, day)

session_names = ['of1', 'vr', 'of2']

output_folders = [f"{project_path}data/M{mouse}_D{day}/{str(raw_recording).split('/')[-1]}/" for raw_recording in raw_recordings]

print(raw_recordings)
print(output_folders)

for raw_recording_path, output_folder in zip(raw_recordings, output_folders):
    print("first raw path: ", raw_recording_path)
    recording = si.read_openephys(Path(raw_recording_path), load_sync_channel=True)
    recording.channel_slice(channel_ids=["CH_SYNC"]).save_to_zarr(folder = output_folder + "channel_sync")
    recording = si.read_openephys(raw_recording_path)
    assert recording.get_num_channels() == 384, "Something wrong with channel indexing while making zarr file"
    recording.save_to_zarr(folder = output_folder + "recording")




