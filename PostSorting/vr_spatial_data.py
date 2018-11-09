import numpy as npimport osimport pandas as pdimport open_ephys_IOimport PostSorting.parametersimport mathimport gcfrom scipy import statsimport PostSorting.vr_stop_analysisimport PostSorting.vr_make_plots# finds time animal spent in each location bin for each trialdef calculate_binned_dwell_time(raw_position_data,processed_position_data):    print('Calculating binned dwell time...')    dwell_rate_map = pd.DataFrame(columns=['trial_number','bin_count', 'dwell_time_ms'])    bin_size_cm,number_of_bins,bins = PostSorting.vr_stop_analysis.get_bin_size(raw_position_data)    number_of_trials = raw_position_data.trial_number.max() # total number of trials    trials = np.array(raw_position_data['trial_number'], dtype=np.uint8)    locations = np.array(raw_position_data['x_position_cm'], dtype=np.float16)    dwell_time_per_sample = np.array(raw_position_data['dwell_time_ms'], dtype=np.float16)    for t in range(1,int(number_of_trials)+1):        trial_locations = np.take(locations, np.where(trials == t)[0])        for loc in range(int(number_of_bins)):            time_in_bin = sum(dwell_time_per_sample[np.where(np.logical_and(trial_locations > loc, trial_locations <= (loc+1)))])            if time_in_bin == 0: # this only happens if the session is started/stopped in the middle of a trial                dwell_rate_map = dwell_rate_map.append({"trial_number": int(t), "bin_count": int(loc),  "dwell_time_ms":  float(0.001)}, ignore_index=True)            else:                dwell_rate_map = dwell_rate_map.append({"trial_number": int(t), "bin_count": int(loc),  "dwell_time_ms":  (time_in_bin)}, ignore_index=True)    processed_position_data['binned_time_ms'] = dwell_rate_map['dwell_time_ms']    return processed_position_datadef calculate_binned_dwell_time_test(raw_position_data,processed_position_data):    print('Calculating binned dwell time...')    dwell_rate_map = pd.DataFrame(columns=['trial_number','bin_count', 'dwell_time_ms'])    bin_size_cm,number_of_bins,bins = PostSorting.vr_stop_analysis.get_bin_size(raw_position_data)    number_of_trials = raw_position_data.trial_number.max() # total number of trials    trials = np.array(raw_position_data['trial_number'])    locations = np.array(raw_position_data['x_position_cm'])    dwell_time_per_sample = np.array(raw_position_data['dwell_time_ms'])    for t in range(1,int(number_of_trials)+1):        trial_locations = np.take(locations, np.where(trials == t)[0])        for loc in range(int(number_of_bins)):            time_start_of_bin = dwell_time_per_sample[np.where(trial_locations == loc)]            time_end_of_bin = dwell_time_per_sample[np.where(trial_locations == loc+1)]            time_in_bin = time_end_of_bin - time_start_of_bin            if time_in_bin == 0: # this only happens if the session is started/stopped in the middle of a trial                dwell_rate_map = dwell_rate_map.append({"trial_number": int(t), "bin_count": int(loc),  "dwell_time_ms":  float(0.001)}, ignore_index=True)            else:                dwell_rate_map = dwell_rate_map.append({"trial_number": int(t), "bin_count": int(loc),  "dwell_time_ms":  (time_in_bin)}, ignore_index=True)    processed_position_data['binned_time_ms_test'] = dwell_rate_map['dwell_time_ms']    #raw_position_data = raw_position_data.drop(['dwell_time_ms'], axis=1)    return processed_position_datadef calculate_binned_time_over_trials(position_data):    print('Calculating binned dwell time...')    dwell_rate_map = pd.DataFrame(columns=['bin_count', 'dwell_time_ms'])    bin_size_cm,number_of_bins,bins = PostSorting.vr_stop_analysis.get_bin_size(position_data)    number_of_trials = position_data.trial_number.max() # total number of trials    locations = np.array(position_data['x_position_cm'].tolist())    dwell_time_per_sample = np.array(position_data['dwell_time_ms'].tolist())    for loc in range(int(number_of_bins)):        time_in_bin = sum(dwell_time_per_sample[np.where(np.logical_and(locations > loc, locations <= (loc+1)))])/number_of_trials        dwell_rate_map = dwell_rate_map.append({"bin_count": int(loc),  "dwell_time_ms":  float(time_in_bin)}, ignore_index=True)    position_data['binned_time_over_trials_seconds'] = dwell_rate_map['dwell_time_ms']    return position_datadef calculate_total_trial_numbers(raw_position_data,processed_position_data):    print('calculating total trial numbers for trial types')    trial_numbers = np.array(raw_position_data['trial_number'])    trial_type = np.array(raw_position_data['trial_type'])    trial_data=np.transpose(np.vstack((trial_numbers, trial_type)))    beaconed_trials = np.delete(trial_data, np.where(trial_data[:,1]>0),0)    unique_beaconed_trials = np.unique(beaconed_trials[:,0])    nonbeaconed_trials = np.delete(trial_data, np.where(trial_data[:,1]!=1),0)    unique_nonbeaconed_trials = np.unique(nonbeaconed_trials[1:,0])    probe_trials = np.delete(trial_data, np.where(trial_data[:,1]!=2),0)    unique_probe_trials = np.unique(probe_trials[1:,0])    processed_position_data.at[0,'beaconed_total_trial_number'] = len(unique_beaconed_trials)    processed_position_data.at[0,'nonbeaconed_total_trial_number'] = len(unique_nonbeaconed_trials)    processed_position_data.at[0,'probe_total_trial_number'] = len(unique_probe_trials)    return processed_position_data"""calculates speed for each location bin (0-200cm) across all trialsinputs:    position_data : pandas dataframe containing position information for mouse during that session    outputs:    position_data : with additional column added for processed data"""def calculate_binned_speed(raw_position_data,processed_position_data):    print('calculate binned speed...')    bin_size_cm,number_of_bins, bins = PostSorting.vr_stop_analysis.get_bin_size(raw_position_data)    number_of_trials = raw_position_data.trial_number.max() # total number of trials    speed_ms = np.array(raw_position_data['speed_per200ms'])    locations = np.array(raw_position_data['x_position_cm'])    speed = []    for loc in range(int(number_of_bins)):        speed_in_bin = (np.mean(speed_ms[np.where(np.logical_and(locations > loc, locations <= (loc+1)))]))/number_of_trials        speed = np.append(speed,speed_in_bin)    processed_position_data['binned_speed_ms'] = pd.Series(speed)    return processed_position_datadef clean_spatial_dataframe(position_data):    position_data.drop(['time_seconds'], axis='columns', inplace=True, errors='ignore')    position_data.drop(['first_series_location_cm'], axis='columns', inplace=True, errors='ignore')    position_data.drop(['first_series_trial_number'], axis='columns', inplace=True, errors='ignore')    position_data.drop(['first_series_trial_type'], axis='columns', inplace=True, errors='ignore')    position_data.drop(['new_trial_indices'], axis='columns', inplace=True, errors='ignore')    return position_datadef process_position_data(raw_position_data, prm):    #position_data = pd.DataFrame(columns=['binned_speed_ms', 'binned_time_ms', 'stop_location_cm', 'stop_trial_number', 'stop_trial_type', 'first_series_location_cm', 'first_series_trial_number', 'first_series_trial_type', 'rewarded_stop_locations', 'rewarded_trials'])    processed_position_data = pd.DataFrame()    processed_position_data = calculate_binned_dwell_time(raw_position_data,processed_position_data)    #processed_position_data = calculate_binned_dwell_time_test(raw_position_data,processed_position_data)    processed_position_data = calculate_binned_speed(raw_position_data,processed_position_data)    processed_position_data = PostSorting.vr_stop_analysis.process_stops(raw_position_data,processed_position_data, prm)    gc.collect()    prm.set_total_length_sampling_points(raw_position_data.time_seconds.values[-1])  # seconds    #position_data = clean_spatial_dataframe(processed_position_data)    raw_position_data.drop(['time_seconds'], axis='columns', inplace=True, errors='ignore')    raw_position_data.drop(['dwell_time_seconds'], axis='columns', inplace=True, errors='ignore')    raw_position_data.drop(['velocity'], axis='columns', inplace=True, errors='ignore')    raw_position_data.drop(['new_trial_indices'], axis='columns', inplace=True, errors='ignore')    return raw_position_data, processed_position_data#  for testingdef main():    print('-------------------------------------------------------------')    params = PostSorting.parameters.Parameters()    recording_folder = 'C:/Users/s1466507/Documents/Ephys/test_overall_analysis/M5_2018-03-06_15-34-44_of'    vr_spatial_data = process_position_data(recording_folder)if __name__ == '__main__':    main()