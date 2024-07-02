import pandas as pd
import settings
from P2_PostProcess.OpenField.rate_map import calculate_rate_maps
from P2_PostProcess.OpenField.Scores.spatial_information import calculate_spatial_information_scores
from P2_PostProcess.OpenField.Scores.half_session import calculate_half_session_stability_scores
from P2_PostProcess.OpenField.Scores.border import calculate_border_scores
from P2_PostProcess.OpenField.Scores.grid import calculate_grid_scores
from P2_PostProcess.OpenField.Scores.head_direction import calculate_head_direction_scores
from P2_PostProcess.OpenField.Scores.speed import calculate_speed_scores

def calculate_corresponding_indices(spike_data, spatial_data, sampling_rate_ephys = settings.sampling_rate):
    avg_sampling_rate_bonsai = float(1 / spatial_data['synced_time'].diff().mean())
    sampling_rate_rate = sampling_rate_ephys / avg_sampling_rate_bonsai

    #remove firing times outside synced time
    clipped_firing_times = []
    bonsai_indices_clusters = []
    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        cluster_df = spike_data[(spike_data.cluster_id == cluster_id)] # dataframe for that cluster
        firing_times = cluster_df["firing_times"].iloc[0]
        firing_times = firing_times[firing_times/sampling_rate_ephys < spatial_data['synced_time'].max()]
        bonsai_indices = firing_times/sampling_rate_rate

        #filter the firing times to remove spikes within the rounding error of the bonsai indexing
        firing_times = firing_times[bonsai_indices.round(0) < len(spatial_data)]
        bonsai_indices = firing_times/sampling_rate_rate
        clipped_firing_times.append(firing_times)
        bonsai_indices_clusters.append(bonsai_indices)

    spike_data["firing_times"] = clipped_firing_times
    spike_data['bonsai_indices'] = bonsai_indices_clusters
    return spike_data


def find_firing_location_indices(spike_data, spatial_data):
    spike_data = calculate_corresponding_indices(spike_data, spatial_data)
    spatial_firing = pd.DataFrame(columns=['position_x', 'position_x_pixels', 'position_y', 'position_y_pixels', 'hd', 'speed'])

    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        cluster_df = spike_data[(spike_data.cluster_id == cluster_id)] # dataframe for that cluster
        bonsai_indices_cluster = cluster_df.bonsai_indices.iloc[0]
        bonsai_indices_cluster_round = bonsai_indices_cluster.round(0)
        spatial_firing = spatial_firing.append({
            "position_x": list(spatial_data.position_x[bonsai_indices_cluster_round]),
            "position_x_pixels": list(spatial_data.position_x_pixels[bonsai_indices_cluster_round]),
            "position_y":  list(spatial_data.position_y[bonsai_indices_cluster_round]),
            "position_y_pixels":  list(spatial_data.position_y_pixels[bonsai_indices_cluster_round]),
            "hd": list(spatial_data.hd[bonsai_indices_cluster_round]),
            "speed": list(spatial_data.speed[bonsai_indices_cluster_round])}, ignore_index=True)

    spike_data['position_x'] = spatial_firing.position_x.values
    spike_data['position_x_pixels'] = spatial_firing.position_x_pixels.values
    spike_data['position_y'] = spatial_firing.position_y.values
    spike_data['position_y_pixels'] = spatial_firing.position_y_pixels.values
    spike_data['hd'] = spatial_firing.hd.values
    spike_data['speed'] = spatial_firing.speed.values
    spike_data = spike_data.drop(['bonsai_indices'], axis=1)
    return spike_data


def add_spatial_variables(spike_data, spatial_data):
    """
    :param spike_data: data frame containing firing times where each row is a neuron
    :param spatial_data: data frame containing position of animal (x, y, hd, time)
    :return: spike_data: updated dataframe containing firing times where each row is a neuron
    with rate maps, classic open field scores and metrics
    """
    spike_data = find_firing_location_indices(spike_data, spatial_data)
    spike_data = calculate_rate_maps(spike_data, spatial_data)
    return spike_data

def add_scores(spike_data, spatial_data, position_heat_map):
    """
    :param spike_data: data frame containing firing times where each row is a neuron
    :param spatial_data: data frame containing position of animal (x, y, hd, time)
    :param position_heat_map: 2D numpy array containing the proportion time spent in bin
    :return: spike_data: updated dataframe containing firing times where each row is a neuron
    with rate maps
    """
    spike_data = calculate_spatial_information_scores(spike_data, position_heat_map)
    spike_data = calculate_head_direction_scores(spike_data, spatial_data)
    spike_data = calculate_border_scores(spike_data)
    spike_data = calculate_grid_scores(spike_data)
    #spike_data = calculate_half_session_stability_scores(spike_data, spatial_data)
    spike_data = calculate_speed_scores(spike_data, spatial_data)
    return spike_data