#  This script is to perform overall analysis on multiple days recorded from a group of animals
import Published_analysis.NatComms_LocalGridHD_2020.compare_whitening_methods
import Published_analysis.NatComms_LocalGridHD_2020.false_positives
import Published_analysis.NatComms_LocalGridHD_2020.population_plots
import Published_analysis.NatComms_LocalGridHD_2020.describe_dataset
import Published_analysis.NatComms_LocalGridHD_2020.recording_location
import pandas as pd
import Published_analysis.NatComms_LocalGridHD_2020.overall_params
import Published_analysis.NatComms_LocalGridHD_2020.population_plots

path_to_data = 'C:/Users/s1466507/Documents/Ephys/recordings/'
save_output_path = 'C:/Users/s1466507/Documents/Ephys/overall_figures/'
false_positives_path = path_to_data + 'false_positives_all.txt'

params = Published_analysis.NatComms_LocalGridHD_2020.overall_params.OverallParameters()


def initialize_parameters():
    params.set_isolation(0.9)
    params.set_noise_overlap(0.05)
    params.set_snr(1)

    params.set_path_to_data('C:/Users/s1466507/Documents/Ephys/recordings/')
    params.set_save_output_path('C:/Users/s1466507/Documents/Ephys/overall_figures/')
    params.set_false_positives_path_all(path_to_data + 'false_positives_all.txt')
    params.set_false_positives_path_separate(path_to_data + 'false_positives_separate.txt')


def run_analyses():
    initialize_parameters()
    Published_analysis.NatComms_LocalGridHD_2020.compare_whitening_methods.compare_whitening(params)

    spike_data_frame = pd.read_csv(path_to_data + 'data_all.csv')  # reads csv, puts it in df
    accepted_clusters = Published_analysis.NatComms_LocalGridHD_2020.false_positives.get_accepted_clusters(spike_data_frame, false_positives_path)
    accepted_clusters = Published_analysis.NatComms_LocalGridHD_2020.recording_location.add_histology_results(accepted_clusters, path_to_data)
    # NatComms_LocalGridHD_2020.read_snippet_data.analyze_snippets(spike_data_frame, path_to_data, save_output_path)

    Published_analysis.NatComms_LocalGridHD_2020.describe_dataset.describe_dataset(accepted_clusters)
    Published_analysis.NatComms_LocalGridHD_2020.describe_dataset.plot_good_cells_per_day(accepted_clusters)

    Published_analysis.NatComms_LocalGridHD_2020.population_plots.plot_all(accepted_clusters, save_output_path)



   #  print(good_light_responsive[["id","cluster","animal", "goodcluster", "lightscoreP"]])


    #snippets = get_snippets('M0_2017-11-21_15-52-53/')  # I will use the folder name as an ID here once it's added to the spreadsheet



def main():
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')
    run_analyses()


if __name__ == '__main__':
    main()