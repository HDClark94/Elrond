import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import spikeinterface.full as si

from Elrond.P2_PostProcess.Shared.traces import *
from Elrond.P2_PostProcess.Shared.plotting import make_pdf_from_recording

def process(recording_path, processed_folder_name, **kwargs):
    #makes plots and/or dataframes and add them to the processed_recording folder

    # plot tracess for spikes and channels
    make_spike_trace_summary(recording_path, processed_folder_name)
    #make_channel_trace_summary(recording_path, processed_folder_name, filtered=False)
    #make_channel_trace_summary(recording_path, processed_folder_name, filtered=True)
    if "plot_lfp" in kwargs.keys() and kwargs["plot_lfp"] == True:
        make_lfp_trace_summary(recording_path, processed_folder_name)

    make_pdf_from_recording(recording_path, processed_folder_name)
    return

