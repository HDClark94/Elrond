from Helpers.upload_download import *
import subprocess
recording_path = os.environ['RECORDING_PATH']
local_scratch_path = os.environ['LOCAL_SCRATCH_PATH']

import sys
print(sys.version)

recording_forepath = os.path.dirname(recording_path)
recordings_to_download = get_recordings_to_sort(recording_path=recording_path, local_path=local_scratch_path, concat_sort=True)

for recording_to_download in recordings_to_download:
    recording_to_download = os.path.dirname(recording_forepath) + "/" + os.path.basename(recording_to_download)

    SOURCE = recording_to_download
    DESTINATION = local_scratch_path+"/"
    print("SOURCE = ", SOURCE)
    print("DESTINATION = ", DESTINATION)
    subprocess.check_call(f'rsync -r {SOURCE} {DESTINATION}', shell=True)

