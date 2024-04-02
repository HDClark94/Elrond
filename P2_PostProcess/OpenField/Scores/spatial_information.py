import numpy as np

def calculate_spatial_information_scores(spike_data):
    '''
    Calculates the spatial information score in bits per spike as in Skaggs et al.,
    1996, 1993).

    To estimate the spatial information contained in the
    firing rate of each cell we used Ispike and Isec – the standard
    approaches used for selecting place cells (Skaggs et al.,
    1996, 1993). We computed the Isec metric from the average firing rate (over trials) in
    the space bins using the following definition:

    Isec = sum(Pj*λj*log2(λj/λ))

    where λj is the mean firing rate in the j-th space bin and Pj
    the occupancy ratio of the bin (in other words, the probability of finding
    the animal in that bin), while λ is the overall
    mean firing rate of the cell. The Ispike metric is a normalization of Isec,
    defined as:

    Ispike = Isec / λ

    This normalization yields values in bits per spike,
    while Isec is in bits per second.
    '''

    spatial_information_Isec_scores = []
    spatial_information_Ispike_scores = []
    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        cluster_df = spike_data[(spike_data.cluster_id == cluster_id)] # dataframe for that cluster

        mean_firing_rate = cluster_df.iloc[0]["mean_firing_rate"] # λ
        firing_rate_map = cluster_df.iloc[0]["firing_maps"] # λj
        occupancy_probability_map = cluster_df.iloc[0]["occupancy_maps"] # Pj
        occupancy_probability_map[np.isnan(occupancy_probability_map)] = 0
        occupancy_probability_map = occupancy_probability_map/np.sum(occupancy_probability_map) # Pj

        if mean_firing_rate > 0:
            Isec = np.sum(occupancy_probability_map * firing_rate_map *
                          np.log2((firing_rate_map / mean_firing_rate) + 0.0001))
            Ispike = Isec / mean_firing_rate
        else:
            Isec = np.nan
            Ispike = np.nan

        if (Ispike < 0):
            print("spatial information shouldn't be negative!")

        spatial_information_Isec_scores.append(Isec)
        spatial_information_Ispike_scores.append(Ispike)

    spike_data["spatial_information_Isec_score"] = spatial_information_Isec_scores
    spike_data["spatial_information_Ispike_score"] = spatial_information_Ispike_scores
    return spike_data