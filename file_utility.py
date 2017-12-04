import glob
import os


def find_the_file(file_path, pattern, type):
    name = None
    file_found = True
    file_name = None

    file_counter = 0
    for name in glob.glob(file_path + pattern):
        file_counter += 1
        pass

    if file_counter > 1:
        print('There are more than one ' + type + ' files in this folder. This is may not be okay.')

    if name is not None:
        file_name = name.rsplit('\\', 1)[1]
    else:
        print('The '+ type + ' file(such as ' + pattern + ' )is not here, or it has an unusual name.')

        file_found = False

    return file_name, file_found


def create_behaviour_folder_structure(prm):
    movement_path = prm.get_filepath() + 'Behaviour'
    prm.set_behaviour_path(movement_path)

    data_path = movement_path + '\\Data'
    prm.set_behaviour_data_path(data_path)

    analysis_path = movement_path + '\\Analysis'
    prm.set_behaviour_analysis_path(analysis_path)

    if os.path.exists(movement_path) is False:
        print('Behavioural data will be saved in {}.'.format(movement_path))
        os.makedirs(movement_path)
        os.makedirs(data_path)
        os.makedirs(analysis_path)


def create_ephys_folder_structure(prm):
    ephys_path = prm.get_filepath() + 'Electrophysiology'
    prm.set_ephys_path(ephys_path)

    spike_path = ephys_path + '\\Spike_sorting'
    prm.set_spike_path(spike_path)

    sorting_t1_path = spike_path + '\\t1_' + prm.get_date()
    sorting_t2_path = spike_path + '\\t2_' + prm.get_date()
    sorting_t3_path = spike_path + '\\t3_' + prm.get_date()
    sorting_t4_path = spike_path + '\\t4_' + prm.get_date()

    sorting_t1_path_continuous = spike_path + '\\t1_' + prm.get_date() + '_continuous'
    sorting_t2_path_continuous = spike_path + '\\t2_' + prm.get_date() + '_continuous'
    sorting_t3_path_continuous = spike_path + '\\t3_' + prm.get_date() + '_continuous'
    sorting_t4_path_continuous = spike_path + '\\t4_' + prm.get_date() + '_continuous'

    analysis_path = ephys_path + '\\Analysis'
    prm.set_ephys_analysis_path(analysis_path)

    data_path = ephys_path + '\\Data'
    prm.set_ephys_data_path(data_path)

    if os.path.exists(ephys_path) is False:
        os.makedirs(ephys_path)
        os.makedirs(spike_path)
        os.makedirs(sorting_t1_path)
        os.makedirs(sorting_t2_path)
        os.makedirs(sorting_t3_path)
        os.makedirs(sorting_t4_path)
        os.makedirs(sorting_t1_path_continuous)
        os.makedirs(sorting_t2_path_continuous)
        os.makedirs(sorting_t3_path_continuous)
        os.makedirs(sorting_t4_path_continuous)

        os.makedirs(analysis_path)
        os.makedirs(data_path)


def create_folder_structure(prm):
    create_behaviour_folder_structure(prm)
    create_ephys_folder_structure(prm)


