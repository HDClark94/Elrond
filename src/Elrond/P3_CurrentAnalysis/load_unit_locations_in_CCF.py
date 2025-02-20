import os
import math
import pandas as pd
import spikeinterface.full as si
# read the matlab files
import numpy as np
import scipy.io
import pandas as pd

def read_probe_mat(probe_locs_path):
    mat = scipy.io.loadmat(probe_locs_path)
    probe_locs = np.array(mat['probe_locs'])
    print(probe_locs)
    return probe_locs


# Function to convert stereotaxic coordinates to ABA CCF
# SC is an array with stereotaxic coordinates to be transformed
# Returns an array containing corresponding CCF coordinates in μm
# Conversion is from this post, which explains the opposite transformation: https://community.brain-map.org/t/how-to-transform-ccf-x-y-z-coordinates-into-stereotactic-coordinates/1858/3
# Warning: this is very approximate
# Warning: the X, Y, Z schematic at the top of the linked post is incorrect, scroll down for correct one.
def StereoToCCF(SC = np.array([1,1,1]), angle = -0.0873):
    # Stretch
    stretch = SC/np.array([1,0.9434,1])
    # Rotate
    rotate = np.array([(stretch[0] * math.cos(angle) - stretch[1] * math.sin(angle)),
                       (stretch[0] * math.sin(angle) + stretch[1] * math.cos(angle)),
                       stretch[2]])
    #Translate
    trans = rotate + np.array([5400, 440, 5700])
    return(trans)

def CCFToStereo(CCF = np.array([1,1,1]), angle = 0.0873):
    #Translate
    trans = CCF - np.array([5400, 440, 5700])
    # Rotate
    rotate = np.array([(trans[0] * math.cos(angle) - trans[1] * math.sin(angle)),
                       (trans[0] * math.sin(angle) + trans[1] * math.cos(angle)),
                       trans[2]])
    # Stretch
    stretch = rotate*np.array([1,0.9434,1])
    return(stretch)


def load_sorting_analzyers(project_path, mouse):
    # get sorting analyzer and unit locations
    day_paths = [f.path for f in os.scandir(f"{project_path}{mouse}/") if f.is_dir()]
    clusters = pd.DataFrame()
    for day_path in day_paths:
        sorting_analyzer_path = f"{day_path}/full/kilosort4/kilosort4_sa"
        of_spikes_path = f"{day_path}/of1/kilosort4/spikes.pkl"

        if os.path.isdir(sorting_analyzer_path) and os.path.exists(of_spikes_path):
            try:
                sorting_analyzer = si.load_sorting_analyzer(sorting_analyzer_path)
                ulc = sorting_analyzer.get_extension("unit_locations")
                qms = sorting_analyzer.get_extension("quality_metrics")
                unit_locations = ulc.get_data(outputs="by_unit")
                quality_metrics = qms.get_data()
                quality_metrics["cluster_id"] = quality_metrics.index
                spike_data = pd.read_pickle(of_spikes_path)
                spike_data = pd.merge(spike_data, quality_metrics, on="cluster_id")

                spike_data['unit_location_x'] = spike_data.index.map(lambda unit: unit_locations[unit][0])
                spike_data['unit_location_y'] = spike_data.index.map(lambda unit: unit_locations[unit][1])
                spike_data['unit_location_z'] = spike_data.index.map(lambda unit: unit_locations[unit][2])

                spike_data = spike_data[(spike_data["snr"] > 1) & 
                                        (spike_data["mean_firing_rate"] > 0.5) & 
                                        (spike_data["rp_contamination"] < 0.9)]

                clusters = pd.concat([clusters, spike_data], ignore_index=True)
            except:
                continue
    return clusters


def add_clusters(probe_locations_path_list, 
                 probe_borders_table_path_list, 
                 project_path, mouse):
    
    prob_locs_list = []
    for probe_locations_path, probe_borders_table_path in zip(probe_locations_path_list, probe_borders_table_path_list):
        probe_locs = read_probe_mat(probe_locations_path)
        borders_table = pd.read_csv(probe_borders_table_path)
        prob_locs = np.array([[probe_locs[0,0], probe_locs[0,1]], 
                              [probe_locs[2,0], probe_locs[2,1]],
                              [probe_locs[1,0], probe_locs[1,1]]])*10
        prob_locs_list.append(prob_locs)
    prob_locs_list = np.array(prob_locs_list)

    clusters_df = load_sorting_analzyers(project_path, mouse)
    # unit_location_x is ML
    # unit_location_y is DV
    # (0, 0) is the tip of the medial most shank and move +/+ in a lateral/dorsal direction
    # Add to scene

    clusters_X_CCF = []
    clusters_Y_CCF = []
    clusters_Z_CCF = []
    clusters_X_SC = []
    clusters_Y_SC = []
    clusters_Z_SC = []
    clusters_annotations = []
    for index, cluster in clusters_df.iterrows():
        shank_id = cluster['shank_id']
        x_pos = cluster['unit_location_x']
        y_pos = cluster['unit_location_y']
        delta_x = (shank_id*250) - x_pos # 250 microns is the assumed distance between NP2 probes
        
        # get probe locations using shank id
        prob_locs = prob_locs_list[shank_id]
        # translate to SC
        prob_locs[:,0] = CCFToStereo(prob_locs[:,0])
        prob_locs[:,1] = CCFToStereo(prob_locs[:,1])
        # Calculate the direction vector from P2 to P1
        direction_vector = prob_locs[:,0] - prob_locs[:,1]
        # Calculate the unit vector in the direction of P1 to P2
        unit_vector = direction_vector / np.linalg.norm(direction_vector)
        # Calculate the position P3, which is Y1 away from P2 along the direction of the unit vector
        P3 = prob_locs[:,1] + (y_pos * unit_vector)
        clusters_X_CCF.append(StereoToCCF(P3)[0])
        clusters_Y_CCF.append(StereoToCCF(P3)[1])
        clusters_Z_CCF.append(StereoToCCF(P3)[2])
        clusters_X_SC.append(P3[0])
        clusters_Y_SC.append(P3[1])
        clusters_Z_SC.append(P3[2])

    clusters_df['X_CCF'] = clusters_X_CCF
    clusters_df['Y_CCF'] = clusters_Y_CCF
    clusters_df['Z_CCF'] = clusters_Z_CCF
    clusters_df['X_SC'] = clusters_X_SC
    clusters_df['Y_SC'] = clusters_Y_SC
    clusters_df['Z_SC'] = clusters_Z_SC
    clusters_df[['X_CCF','Y_CCF', 'Z_CCF']].to_csv('/home/ubuntu/Elrond/probe_data/'+mouse+'.csv')
    return clusters_df


for Mouse in ['M25']:
    clusters_df = add_clusters(probe_locations_path_list=[f'/home/ubuntu/Elrond/probe_data/{Mouse}_probe_locations_1.mat',
                                                          f'/home/ubuntu/Elrond/probe_data/{Mouse}_probe_locations_2.mat',
                                                          f'/home/ubuntu/Elrond/probe_data/{Mouse}_probe_locations_3.mat',
                                                          f'/home/ubuntu/Elrond/probe_data/{Mouse}_probe_locations_4.mat'],
                                probe_borders_table_path_list=[f'/home/ubuntu/Elrond/probe_data/{Mouse}_probe_border_table_1.csv',
                                                               f'/home/ubuntu/Elrond/probe_data/{Mouse}_probe_border_table_2.csv',
                                                               f'/home/ubuntu/Elrond/probe_data/{Mouse}_probe_border_table_3.csv',
                                                               f'/home/ubuntu/Elrond/probe_data/{Mouse}_probe_border_table_4.csv'],
                               project_path="/mnt/datastore/Chris/Cohort12/derivatives/", 
                               mouse=Mouse)
print("stoppp!")