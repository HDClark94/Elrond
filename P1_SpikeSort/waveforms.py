import spikeinterface.full as si
import settings

def get_waveforms(we, sorters):
    waveforms = []
    for i, sorter in enumerate(sorters):
        cluster_waveforms = []
        for j in sorter.get_unit_ids():
            cluster_waveforms.append(we.get_waveforms_segment(i, j, sparsity=None))
        waveforms.append(cluster_waveforms)
    return waveforms


def extract_waveforms(recording, sorting, recording_path, processed_folder_name):
    recording.annotate(is_filtered=settings.bandpass_filter)
    we = si.extract_waveforms(recording, sorting, folder=recording_path +
                              "/" + processed_folder_name + "/" + settings.sorterName + "/waveforms",
                              ms_before=1, ms_after=2, load_if_exists=False,
                              overwrite=True, return_scaled=False)
    _  = si.compute_spike_amplitudes(we)
    _  = si.compute_principal_components(we, n_components=3, mode='by_channel_global')
    _  = si.compute_correlograms(we)
    qm = si.compute_quality_metrics(we, metric_names=settings.list_of_quality_metrics)
    return we, qm