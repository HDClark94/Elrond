import settings
import numpy as np
import spikeinterface.full as si

def ammend_preprocessing_parameters(params):
    params["whiten"] = settings.whiten
    params["filter"] = not settings.bandpass_filter
    params['num_workers'] = settings.n_sorting_workers
    return params

def preprocess(recording):
    if settings.bandpass_filter:
        recording = si.bandpass_filter(recording, freq_min=settings.bandpass_filter_min, freq_max=settings.bandpass_filter_max)
        recording.annotate(is_filtered=settings.bandpass_filter)
    if settings.common_reference:
        recording = si.common_reference(recording, operator="median")
    return recording

