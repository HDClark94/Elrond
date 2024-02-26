"""
Set paths, flags and parameters for the pipeline
"""
##########
# Folder locations
processed_folder = "/processed"
dead_channel_file_name = "/dead_channels.txt"
temp_storage_path = '/home/ubuntu/to_sort/recordings/tmp'

##########
# Recording setting
sampling_rate = 30000
down_sampled_rate = 1000

#########
# Sorter Configuration
sorterName = 'mountainsort4'
list_of_named_sorters = ['MountainSort', 'mountainsort4','klusta','tridesclous','hdsort','ironclust',
                         'kilosort','kilosort2', 'spykingcircus','herdingspikes','waveclus']

whiten = True
common_reference = True
bandpass_filter = True
bandpass_filter_min = 300 # Hz
bandpass_filter_max = 6000 # Hz
n_sorting_workers = 17

############
# Automatic Curation
list_of_quality_metrics = ['snr','isi_violation','firing_rate', 'presence_ratio', 'amplitude_cutoff',\
                          'isolation_distance', 'l_ratio', 'd_prime', 'nearest_neighbor', 'nn_isolation', 'nn_noise_overlap']

# assign the quality metric name, sign of threshold ("<", ">") and value in a tuple
auto_curation_thresholds = [('isolation_distance', '>', 0.9),
                            ('nn_noise_overlap', '<', 0.05),
                            ('snr', '>', 1),
                            ('firing_rate', '>', 0.0)]

##########
# VR

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

use_vectorised_rate_map_function = True
impose_num_cores = False
fixed_num_cores = 1

##################
# Specific to OpenEphys legacy formats
ttl_pulse_channel =     '100_ADC1.continuous' # all session types
movement_channel =      '100_ADC2.continuous' # vr
first_trial_channel =   '100_ADC4.continuous' # vr
second_trial_channel =  '100_ADC5.continuous' # vr

