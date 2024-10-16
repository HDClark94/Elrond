import matplotlib.pylab as plt
import numpy as np
import Published_analysis.NatComms_LocalGridHD_2020.folder_path_settings
import pandas as pd
import PostSorting.compare_first_and_second_half
import PostSorting.open_field_firing_maps
import PostSorting.open_field_make_plots

import PostSorting.parameters
prm = PostSorting.parameters.Parameters()
prm.set_pixel_ratio(440)


local_path = Published_analysis.NatComms_LocalGridHD_2020.folder_path_settings.get_local_path() + '/half_rate_maps/'


def main():
    server_path_mouse = Published_analysis.NatComms_LocalGridHD_2020.folder_path_settings.get_server_path_mouse()
    example_mouse_df_path = server_path_mouse + 'M12_2018-04-10_14-22-14_of/MountainSort/DataFrames/'
    spatial_firing = pd.read_pickle(example_mouse_df_path + 'spatial_firing.pkl')
    position = pd.read_pickle(example_mouse_df_path + 'position.pkl')
    spike_data_first_half, position_data_first_half = PostSorting.compare_first_and_second_half.get_half_of_the_data_cell(prm, spatial_firing, position, half='first_half')
    spike_data_first_half['cluster_id'] = 1
    spikes_first_half = spike_data_first_half.groupby('cluster_id').agg(lambda x: list(x))
    spikes_first_half['cluster_id'] = 1
    spike_data_second_half, position_data_second_half = PostSorting.compare_first_and_second_half.get_half_of_the_data_cell(prm, spatial_firing, position, half='second_half')
    spike_data_second_half['cluster_id'] = 1
    spike_data_second_half['session_id'] = 'M12_04_10'
    spikes_second_half = spike_data_second_half.groupby('cluster_id').agg(lambda x: list(x))
    spikes_second_half['cluster_id'] = 1
    spike_data_second_half['session_id'] = 'M12_04_10'

    position_heat_map_first, firing_data_spatial_first = PostSorting.open_field_firing_maps.make_firing_field_maps(position_data_first_half, spikes_first_half)
    position_heat_map_second, firing_data_spatial_second = PostSorting.open_field_firing_maps.make_firing_field_maps(position_data_second_half, spikes_second_half)
    firing_data_spatial_first['session_id'] = 'M12_04_10'
    firing_data_spatial_second['session_id'] = 'M12_04_10'
    firing_data_spatial_first['firing_fields'] = [spatial_firing.firing_fields.iloc[0]]
    firing_data_spatial_second['firing_fields'] = [spatial_firing.firing_fields.iloc[0]]
    firing_data_spatial_first['spike_times_in_fields'] = [np.array(spatial_firing.spike_times_in_fields.iloc[0])]
    firing_data_spatial_second['spike_times_in_fields'] = [np.array(spatial_firing.spike_times_in_fields.iloc[0])]


    plt.cla()
    plt.imshow(firing_data_spatial_first.firing_maps.iloc[0])
    plt.savefig(local_path + 'first_half_rate_map.png')
    print('max firing rate: ' + str(max(firing_data_spatial_first.firing_maps.iloc[0].flatten())))

    plt.cla()
    plt.imshow(firing_data_spatial_second.firing_maps.iloc[0])
    plt.savefig(local_path + 'second_half_rate_map.png')
    print('max firing rate: ' + str(max(firing_data_spatial_second.firing_maps.iloc[0].flatten())))

    prm.set_output_path(local_path + 'scatter_first/')
    PostSorting.open_field_make_plots.plot_hd(firing_data_spatial_first, position_data_first_half, prm)
    PostSorting.open_field_make_plots.plot_spikes_on_firing_fields(firing_data_spatial_first, prm)
    prm.set_output_path(local_path + 'scatter_second/')
    PostSorting.open_field_make_plots.plot_hd(firing_data_spatial_second, position_data_second_half, prm)
    PostSorting.open_field_make_plots.plot_spikes_on_firing_fields(firing_data_spatial_second, prm)


if __name__ == '__main__':
    main()