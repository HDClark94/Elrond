# This file defines the default values for many spike sorting things such as: the kwargs we use for the sorters, which preprocessing we apply and which sorting analyzer extensions we compute. All of these can be edited here, or a different choices can be passed in to the main spike sorting functions of Elrond.

# which kwargs to use for each sorter
sorter_kwargs_dict = {
    "herdingspikes": {}, 
    "mountainsort5": {"scheme": "3"},
    "kilosort4": {"do_correction": False},
    "kilosort4_motion_correction": {"do_correction": True},
    "spykingcircus2": {}
}

# which preprocessing steps to apply for each sorter, before sorting and before
# postprocessing.
pp_pipelines_dict = {
    "herdingspikes": {
        "sort": {
            "common_reference": {"operator": "average"},
            "highpass_filter": {},
        },
        "post": {
            "common_reference": {"operator": "average"},
            "highpass_filter": {},
        }
    },
    "kilosort4": {
        "sort": {},
        "post": {
            "common_reference": {"operator": "average"},
            "highpass_filter": {},
        }
    },
    "kilosort4_motion_correction": {
        "sort": {},
        "post": {
            "common_reference": {"operator": "average"},
            "highpass_filter": {},
        }
    },
    "mountainsort5": {
        "sort": {
            "bandpass_filter": {},
            "whiten": {},
        },
        "post": {
            "bandpass_filter": {},
            "whiten": {},
        }
    },
    "spykingcircus2": {
        "sort": {
            "common_reference": {"operator": "median"},
            "highpass_filter": {},
        },
        "post": {
            "common_reference": {"operator": "median"},
            "highpass_filter": {},
        }
    }
}

# which sorting analyzer extensions to compute by default. TODO : remove waveforms when si updates.
default_extensions_dict = {"noise_levels": {}, "random_spikes": {}, "waveforms": {}, "templates": {}, "correlograms": {}, "spike_amplitudes": {}, "template_similarity": {}, "quality_metrics": {}}
