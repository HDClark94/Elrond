import numpy as np
import pandas as pd
import spikeinterface.full as si
from scipy.signal import hilbert

def compute_channel_theta_phase(raw_path, save_path, resample_rate=1000):
    raw = si.read_openephys(raw_path)

    # Compute downsampled LFP
    processed = si.bandpass_filter(
        raw,
        freq_min=6.0,
        freq_max=12.0,
        margin_ms=1500.0,
        filter_order=5,
        dtype="float32",
        add_reflect_padding=True,
    )
    processed = si.phase_shift(processed)
    processed = si.resample(processed, resample_rate=resample_rate, margin_ms=1000)

    # to test
    processed = processed.channel_slice(channel_ids=["CH1"])

    # Hilbert transform
    theta_phase = np.angle(hilbert(processed.get_traces(), axis=0)) + np.pi

    # Save
    pd.DataFrame(theta_phase, index=pd.to_datetime(processed.get_times(), unit="s")).to_pickle(
        save_path + "theta_phase_ch1.pkl"
    )