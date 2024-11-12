import numpy as np
import pandas as pd
import spikeinterface.full as si
from scipy.signal import hilbert
from Elrond.Helpers.upload_download import get_recording_from
from pathlib import Path

import numpy as np
import pandas as pd
import spikeinterface.full as si
from scipy.signal import hilbert


def compute_channel_theta_phase(raw_path, save_path, resample_rate=100, parallel_block_size=32):
    
    Path(save_path).mkdir(exist_ok=True, parents=True)

    raw = get_recording_from(raw_path)
    channel_ids = raw.get_channel_ids()

    # Compute downsampled LFP
    processed_ = si.bandpass_filter(
        raw,
        freq_min=6.0,
        freq_max=12.0,
        margin_ms=1500.0,
        filter_order=5,
        dtype="float32",
        add_reflect_padding=True,
    )
    processed_ = si.phase_shift(processed_)
    processed_ = si.resample(processed_, resample_rate=resample_rate, margin_ms=1000)
    timepoints = processed_.get_times()

    # Apply in blocks of channels
    processed = []
    for i in range(0, len(channel_ids), parallel_block_size):
        print(f"Processing channels {i} to {i + parallel_block_size}")
        processed.append(processed_.get_traces(channel_ids=channel_ids[i : i + parallel_block_size]))
    del processed_
    processed = np.concatenate(processed, axis=1)

    # Hilbert transform
    theta_phase = np.angle(hilbert(processed, axis=0)) + np.pi

    # Save
    pd.DataFrame(theta_phase, index=pd.to_datetime(timepoints, unit="s")).to_pickle(save_path + "theta_phase.pkl")
