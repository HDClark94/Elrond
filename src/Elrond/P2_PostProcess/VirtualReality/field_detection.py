import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from astropy.convolution import convolve, Gaussian1DKernel
from P2_PostProcess.VirtualReality.plotting import get_vmin_vmax, min_max_normalize
import matplotlib.pyplot as plt


def make_field_array(firing_rate_map_by_trial, peaks_indices):
    field_array = np.zeros(len(firing_rate_map_by_trial))
    for i in range(len(peaks_indices)):
        field_array[peaks_indices[i][0]:peaks_indices[i][1]] = i+1
    return field_array.astype(np.int64)

 
def get_peak_indices(firing_rate_map, peaks_i):
    peak_indices =[]
    for j in range(len(peaks_i)):
        peak_index_tuple = find_neighbouring_minima(firing_rate_map, peaks_i[j])
        peak_indices.append(peak_index_tuple)
    return peak_indices


def find_neighbouring_minima(firing_rate_map, local_maximum_idx):
    # walk right
    local_min_right = local_maximum_idx
    local_min_right_found = False
    for i in np.arange(local_maximum_idx, len(firing_rate_map)): #local max to end
        if local_min_right_found == False:
            if np.isnan(firing_rate_map[i]):
                continue
            elif firing_rate_map[i] < firing_rate_map[local_min_right]:
                local_min_right = i
            elif firing_rate_map[i] > firing_rate_map[local_min_right]:
                local_min_right_found = True

    # walk left
    local_min_left = local_maximum_idx
    local_min_left_found = False
    for i in np.arange(0, local_maximum_idx)[::-1]: # local max to start
        if local_min_left_found == False:
            if np.isnan(firing_rate_map[i]):
                continue
            elif firing_rate_map[i] < firing_rate_map[local_min_left]:
                local_min_left = i
            elif firing_rate_map[i] > firing_rate_map[local_min_left]:
                local_min_left_found = True

    return (local_min_left, local_min_right)


def detect_fields(firing_rate_map_by_trial, 
                  gauss_kernel_std=2,
                  extra_smooth_gauss_kernel_std=4,
                  peak_min_distance=20):
    
    firing_rate_map_by_trial_flattened = firing_rate_map_by_trial.flatten()
    gauss_kernel_extra = Gaussian1DKernel(stddev=extra_smooth_gauss_kernel_std)
    gauss_kernel = Gaussian1DKernel(stddev=gauss_kernel_std)
    firing_rate_map_by_trial_flattened_extra_smooth = convolve(firing_rate_map_by_trial_flattened, gauss_kernel_extra)

    peaks_i = find_peaks(firing_rate_map_by_trial_flattened_extra_smooth, distance=peak_min_distance)[0]
    peaks_indices = get_peak_indices(firing_rate_map_by_trial_flattened_extra_smooth, peaks_i)
    return peaks_i, peaks_indices


  
def plot_rate_map_relative_to_field(spike_data, tolerance=20, reorder=False):
    if "firing_times_vr" in list(spike_data):
        fr_col = "firing_times_vr"
    else:
        fr_col = "firing_times"
                    
    for cluster_id in spike_data.cluster_id:
        cluster_spike_data = spike_data[spike_data["cluster_id"] == cluster_id]
        firing_times_cluster = cluster_spike_data[fr_col].iloc[0]
        cluster_firing_maps = np.array(cluster_spike_data['fr_binned_in_space_smoothed'].iloc[0])
        cluster_firing_maps_unsmoothed = np.array(cluster_spike_data['fr_binned_in_space'].iloc[0])

        peaks_i, _ = detect_fields(cluster_firing_maps_unsmoothed)
        
        # calculate the largest distance to the next field
        distance_to_next_peak = []
        for i, peak_i in enumerate(peaks_i):
            if i+1>=len(peaks_i):
                distance_to_next_peak.append(0)
            else:
                distance_to_next_peak.append(peaks_i[i+1]-peaks_i[i])
        distance_to_next_peak= np.array(distance_to_next_peak)+tolerance
        largest_distance_to_next_field = np.nanmax(distance_to_next_peak)
        #print(f"max interfield distance is {largest_distance_to_next_field}")

        l = largest_distance_to_next_field
        new_rate_map = np.zeros((len(peaks_i), l)); new_rate_map[:] = np.nan
        for i, peak_i in enumerate(peaks_i):
            if peaks_i[i]+distance_to_next_peak[i]<len(cluster_firing_maps.flatten()):
                #interfield_distance = peaks_i[i+1] - peaks_i[i]
                #new_rate_map[i, :] = cluster_firing_maps.flatten()[peak_i:peak_i+l]
 
                new_rate_map[i, :distance_to_next_peak[i]] = cluster_firing_maps.flatten()[peaks_i[i]:peaks_i[i]+distance_to_next_peak[i]]
            
        if reorder:
            sorted_indices = np.argsort(distance_to_next_peak)
            new_rate_map = new_rate_map[sorted_indices]

        nan_mask = np.isnan(new_rate_map)
        inf_mask = np.isinf(new_rate_map)
        new_rate_map[nan_mask] = 0
        new_rate_map[inf_mask] = 0
        percentile_99th_display = np.nanpercentile(new_rate_map, 95);
        new_rate_map = min_max_normalize(new_rate_map)
        percentile_99th = np.nanpercentile(new_rate_map, 95); 
        new_rate_map = np.clip(new_rate_map, a_min=0, a_max=percentile_99th)
        vmin, vmax = get_vmin_vmax(new_rate_map)
        new_rate_map[nan_mask] = np.nan

        locations = np.arange(0, len(new_rate_map[0]))
        ordered = np.arange(0, len(new_rate_map), 1)
        X, Y = np.meshgrid(locations, ordered)
        cmap = plt.cm.get_cmap("viridis")
        cmap.set_bad("white")
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.pcolormesh(X, Y, new_rate_map, cmap=cmap, shading="auto", vmin=vmin, vmax=vmax)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.xaxis.set_tick_params(labelbottom=False)
        ax.yaxis.set_tick_params(labelleft=False)
        ax.set_title(f"id: {cluster_id}, n_fields: {len(new_rate_map)}")
        plt.subplots_adjust(hspace=.1, wspace=.1, bottom=None, left=None, right=None, top=None)
        plt.show()

 
def plot_all_interfield_distances(spike_data, tolerance=20, reorder=False):
    if "firing_times_vr" in list(spike_data):
        fr_col = "firing_times_vr"
    else:
        fr_col = "firing_times"

    all_distance_to_next_peak = []                
    for cluster_id in spike_data.cluster_id:
        cluster_spike_data = spike_data[spike_data["cluster_id"] == cluster_id]
        firing_times_cluster = cluster_spike_data[fr_col].iloc[0]
        cluster_firing_maps = np.array(cluster_spike_data['fr_binned_in_space_smoothed'].iloc[0])
        cluster_firing_maps_unsmoothed = np.array(cluster_spike_data['fr_binned_in_space'].iloc[0])

        peaks_i, _ = detect_fields(cluster_firing_maps_unsmoothed)
        
        # calculate the largest distance to the next field
        distance_to_next_peak = []
        for i, peak_i in enumerate(peaks_i):
            if i+1>=len(peaks_i):
                distance_to_next_peak.append(0)
            else:
                distance_to_next_peak.append(peaks_i[i+1]-peaks_i[i])
        distance_to_next_peak= np.array(distance_to_next_peak)+tolerance

        all_distance_to_next_peak.append(distance_to_next_peak)

    nrows = int(np.ceil(np.sqrt(len(spike_data))))
    ncols = nrows; i=0; j=0;
    fig, ax = plt.subplots(ncols=ncols, nrows=nrows, figsize=(20, 20), squeeze=False)
    for m, distance_to_next_peak in enumerate(all_distance_to_next_peak):
        ax[j, i].hist(distance_to_next_peak, bins=30, range=(0,200))
        i+=1
        if i==ncols:
            i=0; j+=1
    for j in range(nrows):
        for i in range(ncols):
            ax[j, i].spines['top'].set_visible(False)
            ax[j, i].spines['right'].set_visible(False)
            ax[j, i].spines['bottom'].set_visible(False)
            ax[j, i].spines['left'].set_visible(False)
            ax[j, i].set_xticks([])
            ax[j, i].set_yticks([])
            ax[j, i].xaxis.set_tick_params(labelbottom=False)
            ax[j, i].yaxis.set_tick_params(labelleft=False)
    plt.subplots_adjust(hspace=.1, wspace=.1, bottom=None, left=None, right=None, top=None)
    plt.show()





def plot_all_rate_map_relative_to_field(spike_data, tolerance=20, reorder=False):
    if "firing_times_vr" in list(spike_data):
        fr_col = "firing_times_vr"
    else:
        fr_col = "firing_times"

    rate_maps = []                
    for cluster_id in spike_data.cluster_id:
        cluster_spike_data = spike_data[spike_data["cluster_id"] == cluster_id]
        firing_times_cluster = cluster_spike_data[fr_col].iloc[0]
        cluster_firing_maps = np.array(cluster_spike_data['fr_binned_in_space_smoothed'].iloc[0])
        cluster_firing_maps_unsmoothed = np.array(cluster_spike_data['fr_binned_in_space'].iloc[0])

        peaks_i, _ = detect_fields(cluster_firing_maps_unsmoothed)
        
        # calculate the largest distance to the next field
        distance_to_next_peak = []
        for i, peak_i in enumerate(peaks_i):
            if i+1>=len(peaks_i):
                distance_to_next_peak.append(0)
            else:
                distance_to_next_peak.append(peaks_i[i+1]-peaks_i[i])
        distance_to_next_peak= np.array(distance_to_next_peak)+tolerance
        largest_distance_to_next_field = np.nanmax(distance_to_next_peak)

        l = largest_distance_to_next_field
        new_rate_map = np.zeros((len(peaks_i), l)); new_rate_map[:] = np.nan
        for i, peak_i in enumerate(peaks_i):
            if peaks_i[i]+distance_to_next_peak[i]<len(cluster_firing_maps.flatten()):
                new_rate_map[i, :distance_to_next_peak[i]] = cluster_firing_maps.flatten()[peaks_i[i]:peaks_i[i]+distance_to_next_peak[i]]
            
        if reorder:
            sorted_indices = np.argsort(distance_to_next_peak)
            new_rate_map = new_rate_map[sorted_indices]

        nan_mask = np.isnan(new_rate_map)
        inf_mask = np.isinf(new_rate_map)
        new_rate_map[nan_mask] = 0
        new_rate_map[inf_mask] = 0
        percentile_99th_display = np.nanpercentile(new_rate_map, 95);
        new_rate_map = min_max_normalize(new_rate_map)
        percentile_99th = np.nanpercentile(new_rate_map, 95); 
        new_rate_map = np.clip(new_rate_map, a_min=0, a_max=percentile_99th)
        vmin, vmax = get_vmin_vmax(new_rate_map)
        new_rate_map[nan_mask] = np.nan
        rate_maps.append(new_rate_map)

    cmap = plt.cm.get_cmap("viridis")
    cmap.set_bad("white")
    nrows = int(np.ceil(np.sqrt(len(spike_data))))
    ncols = nrows; i=0; j=0;
    fig, ax = plt.subplots(ncols=ncols, nrows=nrows, figsize=(20, 20), squeeze=False)
    for m, rate_map in enumerate(rate_maps):
        locations = np.arange(0, len(rate_map[0]))
        ordered = np.arange(0, len(rate_map), 1)
        X, Y = np.meshgrid(locations, ordered)
        #vmin, vmax = get_vmin_vmax(rate_map)
        #ax[j, i].pcolormesh(X, Y, rate_map, cmap=cmap, shading="auto", vmin=vmin, vmax=vmax)
        ax[j, i].pcolormesh(X, Y, rate_map, cmap=cmap, shading="auto")
        i+=1
        if i==ncols:
            i=0; j+=1
    for j in range(nrows):
        for i in range(ncols):
            ax[j, i].spines['top'].set_visible(False)
            ax[j, i].spines['right'].set_visible(False)
            ax[j, i].spines['bottom'].set_visible(False)
            ax[j, i].spines['left'].set_visible(False)
            ax[j, i].set_xticks([])
            ax[j, i].set_yticks([])
            ax[j, i].xaxis.set_tick_params(labelbottom=False)
            ax[j, i].yaxis.set_tick_params(labelleft=False)
    plt.subplots_adjust(hspace=.1, wspace=.1, bottom=None, left=None, right=None, top=None)
    plt.show()