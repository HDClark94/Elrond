import numpy as npimport osimport pandas as pdimport open_ephys_IOimport PostSorting.parametersimport itertoolsimport mathimport matplotlib.pylab as pltprm = PostSorting.parameters.Parameters()def keep_first_from_close_series(array, threshold):    num_delete = 1    while num_delete > 0:        diff = np.ediff1d(array, to_begin= threshold + 1)        to_delete = np.where(diff <= threshold)        num_delete = len(to_delete[0])        if num_delete > 0:            array = np.delete(array, to_delete)    return arraydef get_beginning_of_track_positions(position_data):    location = np.array(position_data['x_position_cm']) # Get the raw location from the movement channel    position = 0    beginning_of_track = np.where((location >= position) & (location <= position + 4))    beginning_of_track = np.asanyarray(beginning_of_track)    beginning_plus_one = beginning_of_track + 1    beginning_plus_one = np.asanyarray(beginning_plus_one)    track_beginnings = np.setdiff1d(beginning_of_track, beginning_plus_one)    track_beginnings = keep_first_from_close_series(track_beginnings, 30000)    return track_beginningsdef remove_extra_stops(min_distance, stops):    to_remove = []    for stop in range(len(stops) - 1):        current_stop = stops[stop]        next_stop = stops[stop + 1]        if 0 <= (next_stop - current_stop) <= min_distance:            to_remove.append(stop+1)    filtered_stops = np.asanyarray(stops)    np.delete(filtered_stops, to_remove)    return filtered_stopsdef get_stop_times(position_data):    stops = np.array([])    speed = np.array(position_data['speed_per200ms'].tolist())    threshold = prm.get_stop_threshold()    low_speed = np.where(speed < threshold)    low_speed = np.asanyarray(low_speed)    low_speed_plus_one = low_speed + 1    intersect = np.intersect1d(low_speed, low_speed_plus_one)    stops = np.setdiff1d(low_speed, intersect)    stops = remove_extra_stops(5, stops)    return stopsdef get_stops_on_trials_find_stops(position_data, all_stops, track_beginnings):    print('extracting stops...')    stop_locations = []    stop_trials = []    stop_trial_types = []    location = np.array(position_data['x_position_cm'].tolist())    trial_type = np.array(position_data['trial_type'].tolist())    number_of_trials = int(position_data.trial_number.max()) # total number of trials    all_stops = np.asanyarray(all_stops)    track_beginnings = np.asanyarray(track_beginnings)    for trial in range(number_of_trials - 1):        beginning = track_beginnings[trial]        end = track_beginnings[trial + 1]        all_stops = np.asanyarray(all_stops)        stops_on_trial_indices = (np.where((beginning <= all_stops) & (all_stops <= end)))        stops_on_trial = np.take(all_stops, stops_on_trial_indices)        if len(stops_on_trial) > 0:            stops = np.take(location, stops_on_trial)            trial_types = np.take(trial_type, stops_on_trial)            stop_locations=np.append(stop_locations,stops[0])            stop_trial_types=np.append(stop_trial_types,trial_types[0])            stop_trials=np.append(stop_trials,np.repeat(trial, len(stops[0])))    print('stops extracted')    position_data.at[0,'stop_location_cm'] = stop_locations    position_data.at[0,'stop_trial_number'] = stop_trials    position_data.at[0,'stop_trial_type'] = stop_trial_types    return position_datadef calculate_stops(position_data):    all_stops = get_stop_times(position_data)    track_beginnings = get_beginning_of_track_positions(position_data)    position_data = get_stops_on_trials_find_stops(position_data, all_stops, track_beginnings)    return position_datadef find_first_stop_in_series(position_data):    stop_difference = np.array(position_data['stop_location_cm'].diff().tolist())    first_in_series_indices = np.where(stop_difference > 1)[1]    print('Finding first stops in series')    position_data['first_series_location_cm'] = position_data.stop_location_cm[first_in_series_indices].values    position_data['first_series_trial_number'] = position_data.stop_trial_number[first_in_series_indices].values    position_data['first_series_trial_type'] = position_data.stop_trial_type[first_in_series_indices].values    return position_datadef find_rewarded_positions(position_data):    stop_locations = np.array(position_data['first_series_location_cm'].tolist())    stop_trials = np.array(position_data['first_series_trial_number'].tolist())    rewarded_stop_locations = np.delete(stop_locations, np.where(np.logical_and(stop_locations > 110, stop_locations < 90))[1])    rewarded_trials = np.delete(stop_trials, np.where(np.logical_and(stop_locations > 110, stop_locations < 90))[1])    position_data['rewarded_stop_locations'] = rewarded_stop_locations    position_data['rewarded_trials'] = rewarded_trials    return position_datadef get_bin_size(spatial_data):    bin_size_cm = 1    track_length = spatial_data.x_position_cm.max()    start_of_track = spatial_data.x_position_cm.min()    number_of_bins = (track_length - start_of_track)/bin_size_cm    return bin_size_cm,number_of_binsdef calculate_average_stops(position_data):    stop_locations = np.array(position_data.stop_location_cm.loc[0].tolist())    #stop_trials = np.array(position_data['first_series_trial_number'])    bin_size_cm,number_of_bins = get_bin_size(position_data)    number_of_trials = position_data.trial_number.max() # total number of trials    stops_in_bins = np.zeros((len(range(int(number_of_bins)))))    for loc in range(int(number_of_bins)):        stops_in_bin = len(stop_locations[np.where(np.logical_and(stop_locations > (loc), stop_locations <= (loc+1)))])/number_of_trials        stops_in_bins[loc] = stops_in_bin    position_data.average_stops.iloc[range(int(number_of_bins))] = stops_in_bins    position_data.position_bins.iloc[range(int(number_of_bins))] = range(int(number_of_bins))    return position_datadef process_stops(position_data):    position_data = calculate_stops(position_data)    #position_data = find_rewarded_positions(position_data)    position_data = calculate_average_stops(position_data)    return position_data