import numpy as np 
import Elrond.settings as settings
import pandas as pd 
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.stats.multitest import fdrcorrection
from ..P2_PostProcess.VirtualReality.spatial_firing import add_kinematics, bin_fr_in_space
from joblib import Parallel, delayed
import multiprocessing
import warnings
warnings.filterwarnings("ignore")


def shuffled_times(cluster_firing, position_data, n_shuffles=1000):
    session_id = cluster_firing["session_id"].iloc[0]

    shuffle_firing = pd.DataFrame()
    for i in range(n_shuffles):
        shuffle = cluster_firing.copy()
        shuffle = shuffle[["cluster_id", "session_id", "firing_times", "trial_number"]]
        firing_times = shuffle["firing_times"].to_numpy()[0]/settings.sampling_rate
        trial_numbers = shuffle["trial_number"].to_numpy()[0]
        recording_length_seconds = max(position_data["time_seconds"])-1 

        # generate random index firing time addition independently for each trial spike
        random_firing_additions_by_trial = np.array([])
        for tn in np.unique(trial_numbers):
            tn_n_spikes = len(firing_times[trial_numbers==tn])
            random_firing_additions = np.ones(tn_n_spikes)*np.random.uniform(low=20, high=int(recording_length_seconds)-20)
            random_firing_additions_by_trial = np.append(random_firing_additions_by_trial, random_firing_additions)
        firing_times = firing_times[:len(random_firing_additions_by_trial)] # in cases of any nans 
        shuffled_firing_times = firing_times + random_firing_additions_by_trial
        shuffled_firing_times[shuffled_firing_times >= recording_length_seconds] = shuffled_firing_times[shuffled_firing_times >= recording_length_seconds] - recording_length_seconds # wrap around the firing times that exceed the length of the recording
        shuffled_firing_times = (shuffled_firing_times*settings.sampling_rate).astype(np.int64) # downsample firing times so we can use the position data instead of the raw position data
        shuffle["firing_times"] = [shuffled_firing_times]   
        shuffle_firing = pd.concat([shuffle_firing, shuffle], ignore_index=True)

    shuffle_firing["shuffle_id"] = np.arange(0, n_shuffles)
    shuffle_firing["session_id"] = session_id
    return shuffle_firing


def ramp_score(firing_rate_map_per_trial,
               processed_position_data, 
               trial_type = None, 
               hit_miss_try = None,
               track_region = None):
        # firing_rate_map_per_trial is a (n_trials, n_spatial_bins) numpy array with 1cm bins
        # trial type should be 0,1 or 2
        # hit_miss_try should be "hit", "try" or "run" # sorry for the bad naming
        # track_region should be "outbound", "homebound" or "full"
        if track_region is not None:
            if track_region == "outbound":
                firing_rate_map_per_trial = firing_rate_map_per_trial[:, 30:90]
            elif track_region == "homebound":
                firing_rate_map_per_trial = firing_rate_map_per_trial[:, 110:170]
            else: # full
                firing_rate_map_per_trial = firing_rate_map_per_trial[:, 30:170]   

        if trial_type is not None:
            processed_position_data = processed_position_data[processed_position_data["trial_type"] == trial_type]
        if hit_miss_try is not None:
            processed_position_data = processed_position_data[processed_position_data["hit_miss_try"] == hit_miss_try]
        valid_trial_numbers = np.unique(processed_position_data["trial_number"])
        valid_rate_map = firing_rate_map_per_trial[valid_trial_numbers-1]

        if len(valid_rate_map)>0:
            average_rate_map = np.nanmean(valid_rate_map, axis=0)
            x = sm.add_constant(np.arange(len(average_rate_map)))
            model = sm.OLS(average_rate_map, x) 
            model = model.fit()
            slope = model.params[1]
            pval  = model.pvalues[1]
            return slope, pval
        else:
            return np.nan, np.nan 


def calculate_ramp_scores(spike_data, processed_position_data, 
                          position_data, track_length, save_path,
                          n_shuffles=100, alpha=0.01, save=False):
    # compute shuffles and compute ramp classifications for all permutations 
    # of trial types, trial perforamances and track regions 
    
    model_params = pd.DataFrame() 
    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        cluster_df = spike_data[spike_data["cluster_id"]==cluster_id]

        # shuffle the firing times based on a cyclic shuffling procedure, we add any time between 20 seconds to 10 minutes for all spikes
        shuffle_firing = shuffled_times(cluster_df, position_data, n_shuffles=n_shuffles) 
        shuffle_firing = add_kinematics(shuffle_firing, position_data)
        shuffle_firing = bin_fr_in_space(shuffle_firing, position_data, track_length=track_length, smoothen=False) # dont smoothen as this can introduce false ramps 

        # calculate scores across all trial types, trial performances and track-regions
        for trial_type in [None, 0,1,2]: 
            for hit_miss_try in [None, "hit", "try", "run"]:
                for track_region in [None, "full", "outbound", "homebound"]:
                    #shuffle_params_df = pd.DataFrame(columns=["slope", "pval"])
                    params = ramp_score(np.array(cluster_df["fr_binned_in_space"].iloc[0]),
                                        processed_position_data=processed_position_data,
                                        trial_type=trial_type,
                                        hit_miss_try=hit_miss_try,
                                        track_region=track_region)

                    params_df = pd.DataFrame({'shuffle_id': [-1],
                                              'slope': [params[0]], 
                                              'p_val': [params[1]]})  
                    
                    for sh_i, shuffle in shuffle_firing.iterrows():
                        shuffle_params = ramp_score(np.array(shuffle["fr_binned_in_space"]),
                                                    processed_position_data=processed_position_data,
                                                    trial_type=trial_type,
                                                    hit_miss_try=hit_miss_try,
                                                    track_region=track_region)
                        shuffle_params_df = pd.DataFrame({'shuffle_id': [sh_i],
                                                          'slope': [shuffle_params[0]], 
                                                          'p_val': [shuffle_params[1]]}) 
                       
                        params_df = pd.concat([params_df, shuffle_params_df], ignore_index=True)  
                    
                    #Correct pvalues
                    _, params_df["p_val"] = fdrcorrection(params_df["p_val"])

                    #calculate percentile upper and lower bound limits 
                    lower = np.nanpercentile(params_df["slope"][1:], 5)
                    upper = np.nanpercentile(params_df["slope"][1:], 95)

                    if (params_df["slope"][0] < lower) and (params_df["p_val"][0] < alpha):
                        ramp_class = "-"
                    elif (params_df["slope"][0] > upper) and (params_df["p_val"][0] < alpha):
                        ramp_class = "+"   
                    else:
                        ramp_class = "UN"

                    ramp_df = pd.DataFrame({'cluster_id': [cluster_id],
                                            'trial_type': [trial_type], 
                                            'hit_miss_try': [hit_miss_try],
                                            'track_length': [track_region],
                                            'ramp_class': [ramp_class],}) 
                    model_params = pd.concat([model_params, ramp_df], ignore_index=True)
                    
        print(f'completed ramp classification for cluster {cluster_id}')
    if save:
        model_params.to_pickle(save_path+"ramp_classifications.pkl")
    return model_params


def ramp_parallel(cluster_df, processed_position_data, position_data, track_length, alpha, n_shuffles):
    cluster_df = cluster_df.to_frame().T.reset_index(drop=True)
    cluster_id = cluster_df["cluster_id"].iloc[0]

    # shuffle the firing times based on a cyclic shuffling procedure, we add any time between 20 seconds to 10 minutes for all spikes
    shuffle_firing = shuffled_times(cluster_df, position_data, n_shuffles=n_shuffles) 
    shuffle_firing = add_kinematics(shuffle_firing, position_data)
    shuffle_firing = bin_fr_in_space(shuffle_firing, position_data, 
                                     track_length=track_length, smoothen=False) # dont smoothen as this can introduce false ramps 

    # calculate scores across all trial types, trial performances and track-regions
    model_params = pd.DataFrame() 
    for trial_type in [None, 0,1,2]: 
        for hit_miss_try in [None, "hit", "try", "run"]:
            for track_region in [None, "full", "outbound", "homebound"]:
                #shuffle_params_df = pd.DataFrame(columns=["slope", "pval"])
                params = ramp_score(np.array(cluster_df["fr_binned_in_space"].iloc[0]),
                                    processed_position_data=processed_position_data,
                                    trial_type=trial_type,
                                    hit_miss_try=hit_miss_try,
                                    track_region=track_region)

                params_df = pd.DataFrame({'shuffle_id': [-1],
                                            'slope': [params[0]], 
                                            'p_val': [params[1]]})  
                
                for sh_i, shuffle in shuffle_firing.iterrows():
                    shuffle_params = ramp_score(np.array(shuffle["fr_binned_in_space"]),
                                                processed_position_data=processed_position_data,
                                                trial_type=trial_type,
                                                hit_miss_try=hit_miss_try,
                                                track_region=track_region)
                    shuffle_params_df = pd.DataFrame({'shuffle_id': [sh_i],
                                                        'slope': [shuffle_params[0]], 
                                                        'p_val': [shuffle_params[1]]}) 
                    
                    params_df = pd.concat([params_df, shuffle_params_df], ignore_index=True)  
                
                #Correct pvalues
                _, params_df["p_val"] = fdrcorrection(params_df["p_val"])

                #calculate percentile upper and lower bound limits 
                lower = np.nanpercentile(params_df["slope"][1:], 5)
                upper = np.nanpercentile(params_df["slope"][1:], 95)

                if (params_df["slope"][0] < lower) and (params_df["p_val"][0] < alpha):
                    ramp_class = "-"
                elif (params_df["slope"][0] > upper) and (params_df["p_val"][0] < alpha):
                    ramp_class = "+"   
                else:
                    ramp_class = "UN"

                ramp_df = pd.DataFrame({'cluster_id': [cluster_id],
                                        'trial_type': [trial_type], 
                                        'hit_miss_try': [hit_miss_try],
                                        'track_length': [track_region],
                                        'ramp_class': [ramp_class],}) 
                model_params = pd.concat([model_params, ramp_df], ignore_index=True)
    print(f'completed ramp classification for cluster {cluster_id}') 
    return model_params

                

 
def calculate_ramp_scores_parallel(spike_data, processed_position_data, 
                          position_data, track_length, save_path,
                          n_shuffles=100, alpha=0.01, save=False):

    n_cores = multiprocessing.cpu_count()  
    parallel_model_params = Parallel(n_jobs=n_cores)(delayed(ramp_parallel)(cluster_df, processed_position_data, position_data, track_length, alpha, n_shuffles) for i, cluster_df in spike_data.iterrows())
    model_params = pd.concat(parallel_model_params, ignore_index=True)  
    
    if save: 
        model_params.to_pickle(save_path+"ramp_classifications.pkl")
    return model_params 
