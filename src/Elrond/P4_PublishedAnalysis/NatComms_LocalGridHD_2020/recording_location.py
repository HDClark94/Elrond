import pandas as pd


# read spreadsheet with tetrode location info and add to dataframe
def add_histology_results(spike_df, path_to_data):
    histology_data_frame = pd.read_csv(path_to_data + 'histology_results.csv')  # reads csv, puts it in df
    spike_df = spike_df.merge(histology_data_frame, left_on='animal', right_on='ID', how='outer')
    return spike_df