"""
Set paths, flags and parameters for the pipeline
"""
suppress_warnings = True

##########
# Folder locations
processed_folder_name = "processed"
dead_channel_file_name = "/dead_channels.txt"
temp_storage_path = '/home/ubuntu/to_sort/recordings/tmp'
PIL_fontstyle_path = '/home/ubuntu/Elrond_code/Additional_files/Arial.ttf'
#PIL_fontstyle_path = ('/run/user/1000/gvfs/smb-share:server=cmvm.datastore.ed.ac.uk,share=cmvm/'
#                      'sbms/groups/CDBS_SIDB_storage/NolanLab/ActiveProjects/Harry/Arial.ttf')

##########
# Recording setting
sampling_rate = 30000 # TODO better to inherit from data if available
down_sampled_rate = 1000

#########
# Sorter Configuration
sorterName = 'mountainsort4'
list_of_named_sorters = ['MountainSort', 'mountainsort4','mountainsort5','klusta','tridesclous','hdsort','ironclust',
                         'kilosort','kilosort2', 'kilosort3', 'kilosort4', 'spykingcircus','herdingspikes','waveclus']

whiten = True
common_reference = True
bandpass_filter = True
bandpass_filter_min = 300 # Hz
bandpass_filter_max = 6000 # Hz
n_sorting_workers = 8

############
# Automatic Curation
list_of_quality_metrics = ['snr','isi_violation','firing_rate', 'presence_ratio', 'amplitude_cutoff',\
                          'isolation_distance', 'nearest_neighbor', 'nn_isolation', 'nn_noise_overlap']

# assign the quality metric name, sign of threshold ("<", ">") and value in a tuple
# ('isolation_distance', '>', 0.9) and ('nn_noise_overlap', '<', 0.05) use if pca components are computed

auto_curation_thresholds = [('snr', '>', 1),
                            ('firing_rate', '>', 0.05)]
##########
# VR

vr_deeplabcut_project_path = "/mnt/datastore/Harry/deeplabcut/vr-hc-2024-03-14/"
vr_deeplabcut_pupil_project_path = "/mnt/datastore/Harry/deeplabcut/vr-hc-2024-03-14/"
vr_deeplabcut_licks_project_path = "/mnt/datastore/Harry/deeplabcut/Mouse_Licks-Harry_Clark-2024-09-10/" 
#vr_deeplabcut_project_path = ("/run/user/1000/gvfs/smb-share:server=cmvm.datastore.ed.ac.uk,share=cmvm/"
#                           "sbms/groups/CDBS_SIDB_storage/NolanLab/ActiveProjects/Harry/deeplabcut/vr-hc-2024-03-14")
vr_bin_size_cm = 1
time_bin_size = 0.1 # seconds
guassian_std_for_smoothing_in_time_seconds = 0.2 # seconds
guassian_std_for_smoothing_in_space_cm = 2 # cm
hit_try_run_speed_threshold = 10 # cm/seconds

##########
# Open Field
pixel_ratio = 440
gauss_sd_for_speed_score = 250
open_field_bin_size_cm = 2.5 # cm

use_dlc_for_open_field = True
of_deeplabcut_project_path = "/mnt/datastore/Harry/deeplabcut/openfield_pose-Harry Clark-2024-05-13/"
#of_deeplabcut_project_path = ("/run/user/1000/gvfs/smb-share:server=cmvm.datastore.ed.ac.uk,share=cmvm/"
#                           "sbms/groups/CDBS_SIDB_storage/NolanLab/ActiveProjects/Harry/deeplabcut/openfield_pose-Harry Clark-2024-05-13")

use_vectorised_rate_map_function = True
impose_num_cores = False
fixed_num_cores = 1

##################
# Allen Brain Observatory Visual coding
natural_scenes_image_folder_path = "/mnt/datastore/Harry/Cohort11_april2024/allen_brain_observatory_visual_coding/allen_brain_images"

##################
# Specific to OpenEphys legacy formats
ttl_pulse_channel =     '100_ADC1.continuous' # all session types
movement_channel =      '100_ADC2.continuous' # vr
first_trial_channel =   '100_ADC4.continuous' # vr
second_trial_channel =  '100_ADC5.continuous' # vr

