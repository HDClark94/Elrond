import os
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import spikeinterface.full as si
from Elrond.P2_PostProcess.VirtualReality.spatial_firing import bin_fr_in_space, bin_fr_in_time, add_kinematics

def pull_vr_data(mouse_ids=[], sorter="kilosort4", project_path=""):

    data = pd.DataFrame()
    for mouse in mouse_ids:
        day_paths = [f.path for f in os.scandir(f"{project_path}{mouse}/") if f.is_dir()]

        for day_path in day_paths:
            day = os.path.basename(day_path)
            sorting_analyzer_path = f"{day_path}/full/kilosort4/kilosort4_sa"
            vr_spikes_path = f"{day_path}/vr/kilosort4/spikes.pkl"
            vr_position_data_path = f"{day_path}/vr/position_data.csv"
            vr_processed_position_data_path = f"{day_path}/vr/processed_position_data.pkl"

            if os.path.isdir(sorting_analyzer_path) and os.path.exists(vr_spikes_path) and os.path.exists(vr_position_data_path):
                position_data_df = pd.read_csv(vr_position_data_path) # load position data

                sorting_analyzer = si.load_sorting_analyzer(sorting_analyzer_path) # load curation stats
                ulc = sorting_analyzer.get_extension("unit_locations")
                qms = sorting_analyzer.get_extension("quality_metrics")
                unit_locations = ulc.get_data(outputs="by_unit")
                quality_metrics = qms.get_data()
                quality_metrics["cluster_id"] = quality_metrics.index

                spikes_df = pd.read_pickle(vr_spikes_path) # load spikes
                spikes_df = spikes_df[["session_id", "cluster_id", "firing_times", "mean_firing_rate", "shank_id"]] # reduce to basic set of columns
                spikes_df = pd.merge(spikes_df, quality_metrics, on="cluster_id")
                spikes_df = add_kinematics(spikes_df, position_data_df)
                spikes_df = bin_fr_in_space(spikes_df, position_data_df, track_length=200)
                spikes_df = bin_fr_in_time(spikes_df, position_data_df, track_length=200) 
                spikes_df = spikes_df[(spikes_df["snr"] > 1) & (spikes_df["mean_firing_rate"] > 0.5) & (spikes_df["rp_contamination"] < 0.9)]

                data = pd.concat([data, spikes_df], ignore_index=True) # concat
            else:
                print("couldn't find sorting analyzer or spikes.pkl for recording", day_path)

    return data



def main():

    project_path = "/mnt/datastore/Chris/Cohort12/derivatives/"
    mouse_ids = ["M20", "M21", "M22", "M25", "M26", "M27"]
    sorter = "kilosort4"
    
    M26 = pull_vr_data(mouse_ids=["M26"], sorter="kilosort4", project_path=project_path)
    M26.to_pickle("/mnt/datastore/Harry/SpatialLocationManifolds2025/M26_binned_firing_rates_vr.pkl")

        
    M25 = pull_vr_data(mouse_ids=["M25"], sorter="kilosort4", project_path=project_path)
    M25.to_pickle("/mnt/datastore/Harry/SpatialLocationManifolds2025/M25_binned_firing_rates_vr.pkl")

        
    M27 = pull_vr_data(mouse_ids=["M27"], sorter="kilosort4", project_path=project_path)
    M27.to_pickle("/mnt/datastore/Harry/SpatialLocationManifolds2025/M27_binned_firing_rates_vr.pkl")

    M22 = pull_vr_data(mouse_ids=["M22"], sorter="kilosort4", project_path=project_path)
    M22.to_pickle("/mnt/datastore/Harry/SpatialLocationManifolds2025/M22_binned_firing_rates_vr.pkl")

        
    M20 = pull_vr_data(mouse_ids=["M20"], sorter="kilosort4", project_path=project_path)
    M20.to_pickle("/mnt/datastore/Harry/SpatialLocationManifolds2025/M20_binned_firing_rates_vr.pkl")

        
    M21 = pull_vr_data(mouse_ids=["M21"], sorter="kilosort4", project_path=project_path)
    M21.to_pickle("/mnt/datastore/Harry/SpatialLocationManifolds2025/M21_binned_firing_rates_vr.pkl")


    print("data made")
    
if __name__ == '__main__':
    main()
