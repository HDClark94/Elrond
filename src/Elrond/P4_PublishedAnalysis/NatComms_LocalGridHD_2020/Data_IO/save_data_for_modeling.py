import numpy as np
import os
import pandas as pd


def save_session_data(position, save_path):
    position.to_csv(save_path + 'session.csv')


def save_firing_data(spatial_firing, save_path):
    for index, cell in spatial_firing.iterrows():
        cell_df = pd.DataFrame()
        cell_df['firing_times'] = cell.firing_times
        cell_df['position_x'] = cell.position_x
        cell_df['position_y'] = cell.position_y
        cell_df['position_x_pixels'] = cell.position_x_pixels
        cell_df['position_y_pixels'] = cell.position_y_pixels
        cell_df['hd'] = cell.hd
        if 'speed' in spatial_firing:
            cell_df['speed'] = cell.speed
        cell_df.to_csv(save_path + str(cell.cluster_id) + '_firing_events.csv')


def load_data_frames(path_to_session):
    session_path = path_to_session + 'position.pkl'
    spikes_path = path_to_session + 'spatial_firing.pkl'
    if os.path.exists(session_path):
        session = pd.read_pickle(session_path)
        spikes = pd.read_pickle(spikes_path)
    else:
        return None, None, False
    return session, spikes, True


def process_data():
    save_path = 'C:/Users/s1466507/Dropbox/Edinburgh/grid_fields/for_modeling/'
    session, spikes, done = load_data_frames('//cmvm.datastore.ed.ac.uk/cmvm/sbms/groups/mnolan_NolanLab/ActiveProjects/Klara/Open_field_opto_tagging_p038/M12_2018-04-10_14-22-14_of/MountainSort/DataFrames/')
    if not os.path.exists(save_path + spikes.session_id.iloc[0]):
        os.makedirs(save_path + spikes.session_id.iloc[0])
    save_session_data(session, save_path + spikes.session_id.iloc[0] + '/')
    save_firing_data(spikes, save_path + spikes.session_id.iloc[0] + '/')

    path_to_ventral = '//cmvm.datastore.ed.ac.uk/cmvm/sbms/groups/mnolan_NolanLab/ActiveProjects/Klara/grid_fields/simulated_data/ventral_narrow/'
    tag = 'vn'
    for dir, subdirs, files in os.walk(path_to_ventral):
        for session_name in subdirs:
            session, spikes, done = load_data_frames(dir + session_name + '/')
            if not done:
                continue
            name = save_path + spikes.session_id.iloc[0] + '_' + tag + '_' + session_name + '/'
            if not os.path.exists(name):
                os.makedirs(name)
            save_session_data(session, name)
            save_firing_data(spikes, name)

    path_to_control = '//cmvm.datastore.ed.ac.uk/cmvm/sbms/groups/mnolan_NolanLab/ActiveProjects/Klara/grid_fields/simulated_data/control_narrow/'
    tag = 'cn'
    for dir, subdirs, files in os.walk(path_to_control):
        for session_name in subdirs:
            session, spikes, done = load_data_frames(dir + session_name + '/')
            if not done:
                continue
            name = save_path + spikes.session_id.iloc[0] + '_' + tag + '_' + session_name + '/'
            if not os.path.exists(name):
                os.makedirs(name)
            save_session_data(session, name)
            save_firing_data(spikes, name)


def main():
    process_data()


if __name__ == '__main__':
    main()