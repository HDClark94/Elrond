import pandas as pd
import matplotlib.pylab as plt
import plot_utility

path = 'C:/Users/s1466507/Dropbox/Edinburgh/PhD/thesis/5 firing_properties/fields_correlation.xlsx'


def plot_correlation_coef_hist(correlation_coefs, save_path):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
    fig, ax = plot_utility.style_plot(ax)
    ax.hist(correlation_coefs, color='navy')
    plt.xlabel('Correlation coefficient', fontsize=22)
    plt.ylabel('Number of fields', fontsize=22)
    plt.xlim(-1, 1)
    plt.axvline(x=0, color='red')
    plt.savefig(save_path)
    plt.close()


fields = pd.read_excel(path)
print(fields.head())
significant = (fields.p_value < 0.001)
correlation_coefs = fields[significant]['correlation coef'].values
save_path = path + 'correlation_coef_hist.png'
plot_correlation_coef_hist(correlation_coefs, save_path)


grid_cells = fields['cell type'] == 'grid'
hd_cells = fields['cell type'] == 'hd'
conjunctive_cells = fields['cell type'] == 'conjunctive'
not_classified = fields['cell type'] == 'na'
fields[grid_cells & significant]['correlation coef'].std()

grid_coeffs = fields[grid_cells & significant]['correlation coef'].values
save_path = path + 'correlation_coef_hist_grid.png'
plot_correlation_coef_hist(grid_coeffs, save_path)

grid_coeffs = fields[hd_cells & significant]['correlation coef'].values
save_path = path + 'correlation_coef_hist_hd.png'
plot_correlation_coef_hist(grid_coeffs, save_path)

grid_coeffs = fields[not_classified & significant]['correlation coef'].values
save_path = path + 'correlation_coef_hist_nc.png'
plot_correlation_coef_hist(grid_coeffs, save_path)

grid_coeffs = fields[conjunctive_cells & significant]['correlation coef'].values
save_path = path + 'correlation_coef_hist_conj.png'
plot_correlation_coef_hist(grid_coeffs, save_path)