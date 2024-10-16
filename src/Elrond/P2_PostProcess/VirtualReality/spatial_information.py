import numpy as np

def calculate_spatial_information(spatial_firing, position_data, track_length):
    if "firing_times_vr" in list(spatial_firing):
        fr_col = "firing_times_vr"
    else:
        fr_col = "firing_times"

    position_heatmap = np.zeros(track_length)
    for x in np.arange(track_length):
        bin_occupancy = len(position_data[(position_data["x_position_cm"] > x) &
                                                (position_data["x_position_cm"] <= x+1)])
        position_heatmap[x] = bin_occupancy
    position_heatmap = position_heatmap*np.diff(position_data["time_seconds"])[-1] # convert to real time in seconds
    occupancy_probability_map = position_heatmap/np.sum(position_heatmap) # Pj

    spatial_information_scores_Ispike = []
    spatial_information_scores_Isec = []
    for cluster_index, cluster_id in enumerate(spatial_firing.cluster_id):
        cluster_df = spatial_firing[(spatial_firing.cluster_id == cluster_id)] # dataframe for that cluster

        mean_firing_rate = len(cluster_df.iloc[0][fr_col])/np.sum(len(position_data)*np.diff(position_data["time_seconds"])[-1]) # λ
        spikes, _ = np.histogram(np.array(cluster_df['x_position_cm'].iloc[0]), bins=track_length, range=(0,track_length))
        rates = spikes/position_heatmap
 
        Isec, Ispike = spatial_info(mean_firing_rate, occupancy_probability_map, rates)

        spatial_information_scores_Ispike.append(Ispike)
        spatial_information_scores_Isec.append(Isec)

    spatial_firing["spatial_information_score_Isec"] = spatial_information_scores_Isec
    spatial_firing["spatial_information_score_Ispike"] = spatial_information_scores_Ispike
    return spatial_firing

def spatial_info(mrate, occupancy_probability_map, rates):
    Isec = np.sum(occupancy_probability_map * rates * np.log2((rates / mrate) + 0.0001))
    Ispike = Isec / mrate
    return Isec, Ispike