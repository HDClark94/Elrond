import math_utility
import numpy as np
import pandas as pd
import PostSorting.open_field_spatial_firing
import data_frame_utility


def calculate_heading_direction(position_x, position_y, pad_first_value=True):
    """
    Calculate heading direction of animal based on the central position of the tracking markers.
    Method from:
    https://doi.org/10.1016/j.brainres.2014.10.053

    input : position_x and position_y of the animal (arrays)
            pad_first_value - if True, the first value will be repeated so the output array's shape is the
            same as the input
    output : heading direction of animal
    based on the vector from consecutive samples
    """


    delta_x = np.diff(position_x)
    delta_y = np.diff(position_y)
    heading_direction = np.arctan(delta_y / delta_x)
    rho, heading_direction = math_utility.cart2pol(delta_x, delta_y)

    heading_direction_deg = np.degrees(heading_direction)
    if pad_first_value:
        heading_direction_deg = np.insert(heading_direction_deg, 0, heading_direction_deg[0])

    return heading_direction_deg


def add_heading_direction_to_position_data_frame(position):
    x = position.position_x
    y = position.position_y
    heading_direction = calculate_heading_direction(x, y, pad_first_value=True)
    position['heading_direction'] = heading_direction
    return position


# add heading direction to spatial firing df
def add_heading_direction_to_spatial_firing_data_frame(spatial_firing, position, ephys_sampling_rate):
    if 'heading_direction' not in position:
        position = add_heading_direction_to_position_data_frame(position)

    headings = []
    spatial_firing = PostSorting.open_field_spatial_firing.calculate_corresponding_indices(spatial_firing, position, avg_sampling_rate_open_ephys=ephys_sampling_rate)
    for index, cluster in spatial_firing.iterrows():
        bonsai_indices_cluster_round = cluster.bonsai_indices.round(0)
        heading = list(position.heading_direction[bonsai_indices_cluster_round])
        headings.append(heading)
    spatial_firing['heading_direction'] = headings
    return spatial_firing, position


def add_heading_direction_to_spatial_firing_data_frame_one_cluster(cluster, position, ephys_sampling_rate):
    if 'heading_direction' not in position:
        position = add_heading_direction_to_position_data_frame(position)

    headings = []
    spatial_firing = PostSorting.open_field_spatial_firing.calculate_corresponding_indices(cluster, position, avg_sampling_rate_open_ephys=ephys_sampling_rate)
    bonsai_indices_cluster_round = cluster.bonsai_indices.round(0)
    heading = list(position.heading_direction[bonsai_indices_cluster_round])
    headings.append(heading)
    spatial_firing['heading_direction'] = headings
    return spatial_firing, position


def calculate_corresponding_indices(spike_data, spatial_data, avg_sampling_rate_open_ephys=30000):
    avg_sampling_rate_bonsai = float(1 / spatial_data['synced_time'].diff().mean())
    sampling_rate_rate = avg_sampling_rate_open_ephys / avg_sampling_rate_bonsai
    bonsai_indices_all = (np.array(spike_data.spike_times) / sampling_rate_rate)
    spike_data['bonsai_indices'] = bonsai_indices_all
    return spike_data


def calculate_corresponding_indices_trajectory(spike_data, spatial_data, avg_sampling_rate_open_ephys=30000):
    spike_data['bonsai_indices_trajectory'] = np.array(spike_data.times_session)
    return spike_data


def add_heading_during_spikes_to_field_df2(field, position, ephys_sampling):
    if 'heading_direction' not in position:
        position = add_heading_direction_to_position_data_frame(position)
    avg_sampling_rate_bonsai = round(int(1 / position['synced_time'].diff().median()), -1)
    sampling_rate_rate = ephys_sampling / avg_sampling_rate_bonsai
    field['bonsai_indices'] = np.array(field.spike_times) / sampling_rate_rate
    position['original_indices'] = (position.synced_time * avg_sampling_rate_bonsai).round(0).astype(int)
    bonsai_indices_cluster_round = field.bonsai_indices.round(0)
    headings = []
    for spike in bonsai_indices_cluster_round:
        heading = position.iloc[(position['original_indices'] - spike).abs().argsort()[0]].heading_direction
        headings.append(heading)

    field['heading_direction_in_field_spikes'] = headings
    return field


def add_heading_during_spikes_to_field_df(field, position, ephys_sampling):
    if 'heading_direction' not in position:
        position = add_heading_direction_to_position_data_frame(position)
    fields = calculate_corresponding_indices(field, position, avg_sampling_rate_open_ephys=ephys_sampling)
    bonsai_indices_cluster_round = fields.bonsai_indices.round(0)
    heading = list(position.heading_direction[bonsai_indices_cluster_round])
    fields['heading_direction_in_field_spikes'] = heading
    return fields


def add_heading_from_trajectory_to_field_df2(field, position, ephys_sampling):
    if 'heading_direction' not in position:
        position = add_heading_direction_to_position_data_frame(position)
    avg_sampling_rate_bonsai = round(int(1 / position['synced_time'].diff().median()), -1)

    field['bonsai_indices'] = np.array(field.times_session) * avg_sampling_rate_bonsai
    position['original_indices'] = (position.synced_time * avg_sampling_rate_bonsai).round(0).astype(int)
    bonsai_indices_cluster_round = field.bonsai_indices.round(0)
    headings = []
    for sample in bonsai_indices_cluster_round:
        heading = position.iloc[(position['original_indices'] - sample).abs().argsort()[0]].heading_direction
        headings.append(heading)
    field['heading_direction_in_field_trajectory'] = headings
    return field


def add_heading_from_trajectory_to_field_df(fields, position, ephys_sampling):
    if 'heading_direction' not in position:
        position = add_heading_direction_to_position_data_frame(position)
    headings = []
    fields = calculate_corresponding_indices_trajectory(fields, position)
    bonsai_indices_cluster_round = fields.bonsai_indices_trajectory.round(0)
    heading = list(position.heading_direction[bonsai_indices_cluster_round])
    headings.append(heading)
    fields['heading_direction_in_field_trajectory'] = headings
    return fields


# add heading direction to field df (where each row is data from a firing field - see data_frame_utility
def add_heading_direction_to_fields_frame(fields, position, ephys_sampling):
    if 'heading_direction' not in position:
        position = add_heading_direction_to_position_data_frame(position)
    fields = add_heading_during_spikes_to_field_df(fields, position, ephys_sampling)
    fields = add_heading_from_trajectory_to_field_df(fields, position)
    return fields, position


def main():
    x = [0, 1, 2, 2, 1]
    y = [0, 1, 1, 0, 1]
    heading_direction_deg = calculate_heading_direction(x, y)

    path = 'C:/Users/s1466507/Documents/Ephys/recordings/M5_2018-03-06_15-34-44_of/MountainSort/DataFrames/'
    position_path = path + 'position.pkl'
    position = pd.read_pickle(position_path)
    spatial_firing_path = path + 'spatial_firing.pkl'
    spatial_firing = pd.read_pickle(spatial_firing_path)
    position = add_heading_direction_to_position_data_frame(position)
    # spatial_firing, position = add_heading_direction_to_spatial_firing_data_frame(spatial_firing, position)

    field_df = data_frame_utility.get_field_data_frame(spatial_firing, position)
    field_df, position = add_heading_direction_to_fields_frame(field_df, position)


if __name__ == '__main__':
    main()