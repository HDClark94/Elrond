import numpy as np
import matplotlib.pyplot as plt
import spikeinterface as si
import spikeinterface.extractors as se

from probeinterface.plotting import plot_probe
from probeinterface import get_probe
from probeinterface import Probe, ProbeGroup

import pandas as pd
import os
from neuroconv.utils.dict import load_dict_from_file

import settings

def get_probe_dataframe(number_of_channels):
    if number_of_channels == 16: # presume tetrodes
        geom = pd.read_csv(settings.tetrode_geom_path, header=None).values
        probe = Probe(ndim=2, si_units='um')
        probe.set_contacts(positions=geom, shapes='circle', shape_params={'radius': 5})
        probe.set_device_channel_indices(np.arange(number_of_channels))
        probe_df = probe.to_dataframe()
        probe_df["channel"] = np.arange(1,16+1)
        probe_df["shank_ids"] = 1
        probe_df["probe_index"] = 1

    else: # presume cambridge neurotech P1 probes
        assert number_of_channels%64==0
        probegroup = ProbeGroup()
        n_probes = int(number_of_channels/64)
        device_channel_indices = []
        for i in range(n_probes):
            probe = get_probe('cambridgeneurotech', 'ASSY-236-P-1')
            probe.wiring_to_device('cambridgeneurotech_mini-amp-64', channel_offset=int(i * 64))
            probe.move([i * 2000, 0])  # move the probes far away from eachother
            probe.set_contact_ids(np.array(probe.to_dataframe()["contact_ids"].values, dtype=np.int64) + int(64 * i))
            probegroup.add_probe(probe)
            # TODO IS THIS RIGHT?
            device_channel_indices.extend(probe.device_channel_indices.tolist())

        device_channel_indices = np.array(device_channel_indices)+1
        probe_df = probegroup.to_dataframe()
        probe_df = probe_df.astype({"probe_index": int, "shank_ids": int, "contact_ids": int})
        probe_df["channel"] = device_channel_indices.tolist()
        probe_df["shank_ids"] = (np.asarray(probe_df["shank_ids"])+1).tolist()
        probe_df["probe_index"] = (np.asarray(probe_df["probe_index"])+1).tolist()
    return probe_df


def test_probe_interface(save_path):
    num_channels = 64
    recording, sorting_true = se.toy_example(duration=5, num_channels=num_channels, seed=0, num_segments=4)

    other_probe = get_probe('cambridgeneurotech', 'ASSY-236-P-1')
    print(other_probe)

    other_probe.set_device_channel_indices(np.arange(num_channels))
    recording_4_shanks = recording.set_probe(other_probe, group_mode='by_shank')
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(1, 1, 1)
    plot_probe(recording_4_shanks.get_probe(), ax=ax)
    plt.savefig(save_path+'probe_locations.png', dpi=200)
    plt.close()

def get_recording_probe(recording_path):
    if os.path.exists(recording_path + "/params.yml"):
        params = load_dict_from_file(recording_path + "/params.yml")
        if ('probe_manufacturer' in params) and ('probe_name' in params)\
                and ('n_probes' in params):
            print("I found the name of the recording probe: ", params['probe_name'])
            return params["probe_manufacturer"], params['probe_name'], params['n_probes']

    print("I couldn't find the name of the recording probe in a params.yml, "
          "I will presume its a tetrode recording")
    return "tetrode", "tetrode", 1

def add_tetrode_geometry(recording):
    geometry = np.array([[   0,  0], [  25,  0], [  25, 25], [  0,  25],
                         [200, 200], [225, 200], [225, 225], [200, 225],
                         [400, 400], [425, 400], [425, 425], [400, 425],
                         [600, 600], [625, 600], [625, 625], [600, 625]])

    probe = Probe(ndim=2, si_units='um')
    probe.set_contacts(positions=geometry, shapes='circle', shape_params={'radius': 5})
    probe.set_device_channel_indices(np.arange(len(geometry)))

    recording.set_probe(probe, group_mode="by_shank", in_place=True)
    return recording, probe

def add_probe_geometry(recording, probe_manufacturer, probe_name, n_probes):
    probegroup = ProbeGroup()
    for i in range(n_probes):
        probe = get_probe(manufacturer=probe_manufacturer, probe_name=probe_name)

        if (probe_manufacturer == "cambridgeneurotech") and (probe_name == "ASSY-236-P-1"):
            probe.wiring_to_device('cambridgeneurotech_mini-amp-64', channel_offset=int(i * 64))
            probe.move([i * 2000, 0])
            probegroup.add_probe(probe)
        else:
            print("I don't know how to handle this probe yet!")

    probe_df = probegroup.to_dataframe()

    recording.set_probegroup(probegroup, group_mode="by_shank", in_place=True)
    return recording, probegroup

def add_probe(recording, recording_path):
    probe_manufacturer, probe_name, n_probes = get_recording_probe(recording_path)

    if probe_name == "tetrode":
        recording, probe = add_tetrode_geometry(recording)
    else:
        recording, probe = add_probe_geometry(recording, probe_manufacturer, probe_name, n_probes)
    return recording, probe