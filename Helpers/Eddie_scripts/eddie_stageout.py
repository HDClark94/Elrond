from Helpers.upload_download import *
import subprocess
recording_path = os.environ['RECORDING_PATH']
local_scratch_path = os.environ['LOCAL_SCRATCH_PATH']

recording_forepath = os.path.dirname(recording_path)
recordings_to_upload = get_recordings_to_sort(recording_path=recording_path, local_path=local_scratch_path, concat_sort=True)

for recording_to_upload in recordings_to_upload:
    recording_to_upload = os.path.dirname(recording_forepath) + "/" + os.path.basename(recording_to_upload)

    SOURCE = local_scratch_path + "/" + os.path.basename(recording_to_upload) + "/processed"
    DESTINATION = recording_to_upload + "/"
    subprocess.check_call(f'rsync -rl {SOURCE} {DESTINATION}', shell=True)