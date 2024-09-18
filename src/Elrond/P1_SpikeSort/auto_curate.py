import Elrond.settings as settings

def auto_curation(spike_data):
    """
    :param spike_data: pandas dataframe with each row defined by a unique cluster_id
    :return: spike_data with automatic curation labels
    """
    # check if quality metrics are found which the spike_data
    for (metric, sign, value) in settings.auto_curation_thresholds:
        if metric not in list(spike_data):
            print("I couldn't find ", metric, " in the spike data")

    curation_labels = []
    for cluster_id in spike_data.cluster_id:
        cluster_spike_data = spike_data[spike_data["cluster_id"] == cluster_id].iloc[0]

        label = True # assume True and then check each condition
        for (metric, sign, value) in settings.auto_curation_thresholds:
            if sign == "<":
                if cluster_spike_data[metric] >= value:
                    label = False
            elif sign == ">":
                if cluster_spike_data[metric] <= value:
                    label = False
            else:
                print("Sign is not recognised, options are either > or <")

        curation_labels.append(label)
    spike_data["passed_auto_curation"] = curation_labels
    return spike_data

