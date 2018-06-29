import pandas as pd


def calculate_corresponding_indices(spike_data, spatial_data):
    avg_sampling_rate_bonsai = float(1 / spatial_data['synced_time'].diff().mean())
    avg_sampling_rate_open_ephys = 30000  # Hz
    sampling_rate_rate = avg_sampling_rate_open_ephys/avg_sampling_rate_bonsai
    spike_data['bonsai_indices'] = spike_data.firing_times/sampling_rate_rate
    return spike_data


def find_firing_location_indices(spike_data, spatial_data):
    spike_data = calculate_corresponding_indices(spike_data, spatial_data)
    spatial_firing = pd.DataFrame(columns=['position_x', 'position_y', 'hd'])
    for cluster in range(len(spike_data)):
        bonsai_indices_cluster = spike_data.bonsai_indices[cluster]
        bonsai_indices_cluster_round = bonsai_indices_cluster.round(0)
        spatial_firing = spatial_firing.append({
            "position_x": list(spatial_data.position_x[bonsai_indices_cluster_round]),
            "position_y":  list(spatial_data.position_y[bonsai_indices_cluster_round]),
            "hd": list(spatial_data.hd[bonsai_indices_cluster_round])
        }, ignore_index=True)
    spike_data['position_x'] = spatial_firing.position_x
    spike_data['position_y'] = spatial_firing.position_y
    spike_data['hd'] = spatial_firing.hd
    return spike_data


def add_firing_locations(spike_data, spatial_data):
    spike_data = find_firing_location_indices(spike_data, spatial_data)
    spike_data = spike_data.drop(['bonsai_indices'], axis=1)
    return spike_data


def process_spatial_firing(spike_data, spatial_data):
    spatial_spike_data = add_firing_locations(spike_data, spatial_data)
    return spatial_spike_data
