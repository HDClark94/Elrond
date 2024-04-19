import spikeinterface.full as si
import settings

def get_waveforms(we, sorters):
    waveforms = []
    for i, sorter in enumerate(sorters):
        cluster_waveforms = []
        for id in sorter.get_unit_ids():
            cluster_waveforms.append(we.get_waveforms_segment(i, id, sparsity=None))
        waveforms.append(cluster_waveforms)
    return waveforms


def extract_waveforms(recording, sorting, recording_path, processed_folder_name):
    we = si.extract_waveforms(recording, sorting, folder=recording_path +
                              "/" + processed_folder_name + "/" + settings.sorterName + "/waveforms",
                              ms_before=1, ms_after=2, load_if_exists=False,
                              overwrite=True, return_scaled=False, max_spikes_per_unit=250)
    sparsity = si.compute_sparsity(we, method="radius")
    _  = si.compute_spike_amplitudes(we)
    _  = si.compute_principal_components(we, n_components=3)
    _  = si.compute_correlograms(we)
    _  = si.compute_unit_locations(we)
    _  = si.compute_template_similarity(we)
    qm = si.compute_quality_metrics(we, metric_names=settings.list_of_quality_metrics, sparsity=sparsity, load_if_exists=True, skip_pc_metrics=True)

    ####https://github.com/SpikeInterface/spikeinterface/issues/572
    # sparsity_dict=dict(method="by_property", by_property="group")
    return we, qm