import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import spikeinterface.full as si

from P2_PostProcess.Shared.traces import *

def process(recording_path, processed_folder_name, **kwargs):
    #makes plots and/or dataframes and add them to the processed_recording folder

    make_lfp_trace_summary(recording_path, processed_folder_name)

    # plot tracess for spikes and channels

    #TODO to match the lfp trace summary, take 4 second trace windows and only plot for the 2 seconds inside each window
    make_spike_trace_summary(recording_path, processed_folder_name)
    make_channel_trace_summary(recording_path, processed_folder_name, filtered=False)
    make_channel_trace_summary(recording_path, processed_folder_name, filtered=True)

    make_summary_pdf(recording_path, processed_folder_name)

    return

