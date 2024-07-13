import os
import subprocess
import sys

recording_to_process = sys.argv[1]
# something like /exports/cmvm/datastore/sbms/groups/CDBS_SIDB_storage/NolanLab/ActiveProjects/Harry/Cohort11_april2024/vr/M20_D14_2024-05-13_16-45-26_VR1

LOCAL_SCRATCH_PATH = '/exports/eddie/scratch/hclark3/recordings'
STAGEIN_PATH = '/home/hclark3/Elrond/Helpers/Eddie_scripts/eddie_stagein.sh'
STAGEOUT_PATH = '/home/hclark3/Elrond/Helpers/Eddie_scripts/eddie_stageout.sh'
PIPELINE_PATH = '/home/hclark3/Elrond/Helpers/Eddie_scripts/eddie_pipeline.sh'

download_job_name = os.path.basename(recording_to_process)[:7]+"_DL"
pipeline_job_name = os.path.basename(recording_to_process)[:7]+"_PL"
upload_job_name = os.path.basename(recording_to_process)[:7]+"_UL"


subprocess.check_call(f'qsub -N {download_job_name} '\
         f'-v RECORDING_PATH={recording_to_process} '\
         f'-v LOCAL_SCRATCH_PATH={LOCAL_SCRATCH_PATH} '\
         f'{STAGEIN_PATH}', shell=True)
print("submitted job ", download_job_name)

subprocess.check_call(f'qsub -N {pipeline_job_name} '\
                      f'-hold_jid {download_job_name} '\
                      f'-v RECORDING_PATH={recording_to_process} '\
                      f'-v LOCAL_PATH={LOCAL_SCRATCH_PATH} '\
                      f'-v SPIKESORT={True} '\
                      f'-v UPDATE_FROM_PHY={False} '\
                      f'-v POSTPROCESS={True} '\
                      f'-v CONCATSORT={True} '\
                      f'-v DLC={True} '\
                      f'-v SORTER={"kilosort4"} '\
                      f'{PIPELINE_PATH}', shell=True)
print("submitted job ", pipeline_job_name)

subprocess.check_call(f'qsub -N {upload_job_name} '\
                      f'-hold_jid {pipeline_job_name} '\
                      f'-v RECORDING_PATH={recording_to_process} '\
                      f'-v LOCAL_SCRATCH_PATH={LOCAL_SCRATCH_PATH} '\
                      f'{STAGEOUT_PATH}', shell=True)
print("submitted job ", upload_job_name)


