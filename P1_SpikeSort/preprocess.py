import settings

def ammend_preprocessing_parameters(params):
    params["whiten"] = settings.whiten
    params["filter"] = settings.bandpass_filter
    params["freq_min"] = settings.bandpass_filter_min
    params["freq_max"] = settings.bandpass_filter_max
    params['num_workers'] = settings.n_sorting_workers
    return params

