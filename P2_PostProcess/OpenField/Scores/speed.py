from Helpers import array_utility
import numpy as np
import settings
import scipy.ndimage
import scipy.stats

'''
The speed score is a measure of the correlation between the firing rate of the neuron and the running speed of the
animal. The firing times of the neuron are binned at the same sampling rate as the position data (speed). The resulting
temporal firing histogram is then smoothed with a Gaussian (standard deviation ~250ms). Speed and temporal firing rate
are correlated (Pearson correlation) to obtain the speed score.

Based on: Gois & Tort, 2018, Cell Reports 25, 1872–1884

position : data frame that contains the speed of the animal as a column ('speed').
spatial_firing : data frame that contains the firing times ('firing_times')
sigma : standard deviation for Gaussian filter (sigma = 250 / video_sampling)
sampling_rate_conversion : sampling rate of ephys data relative to seconds. If the firing times are in seconds then this
should be 1.
'''

def calculate_speed_score(spike_data, spatial_data, gauss_sd = settings.gauss_sd_for_speed_score,
                                                    sampling_rate = settings.sampling_rate):
    avg_sampling_rate_video = float(1 / spatial_data['synced_time'].diff().mean())
    sigma = gauss_sd / avg_sampling_rate_video
    speed = scipy.ndimage.filters.gaussian_filter(spatial_data.speed, sigma)
    speed_scores = []
    speed_score_ps = []
    for index, cell in spike_data.iterrows():
        firing_times = cell.firing_times
        firing_hist, edges = np.histogram(firing_times, bins=len(speed), range=(0, max(spatial_data.synced_time)*sampling_rate))
        smooth_hist = scipy.ndimage.filters.gaussian_filter(firing_hist.astype(float), sigma)
        speed, smooth_hist = array_utility.remove_nans_from_both_arrays(speed, smooth_hist)
        speed_score, p = scipy.stats.pearsonr(speed, smooth_hist)
        speed_scores.append(speed_score)
        speed_score_ps.append(p)
    spike_data['speed_score'] = speed_scores
    spike_data['speed_score_p_values'] = speed_score_ps
    return spike_data






