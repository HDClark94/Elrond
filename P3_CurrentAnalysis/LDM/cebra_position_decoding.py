import pandas as pd
import settings as settings

import warnings
import numpy as np
import umap
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE, Isomap
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import interpolate
import joblib as jl
import cebra.datasets
from cebra import CEBRA
import cebra.integrations.plotly

from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
import sklearn.metrics

from matplotlib.collections import LineCollection
from Helpers.array_utility import list_of_list_to_1d_numpy_array

import sklearn.metrics

def encode_1d_to_2d(positions, min_val=0, max_val=200):
    # Calculate the circumference
    circumference = max_val - min_val

    # Normalize positions to fall between 0 and 1
    normalized_positions = (positions - min_val) / circumference

    # Calculate 2D coordinates
    x_positions = np.cos(2 * np.pi * normalized_positions)
    y_positions = np.sin(2 * np.pi * normalized_positions)

    return np.array([x_positions, y_positions]).T


def decode_2d_to_1d(coordinates, min_val=0, max_val=200):
    # Calculate the circumference
    circumference = max_val - min_val

    # Transform 2D coordinates back into angles
    angles = np.arctan2(coordinates[:,1], coordinates[:,0])

    # Normalize angles to fall between 0 and 1
    normalized_angles = (angles % (2 * np.pi)) / (2 * np.pi)

    # Calculate original positions
    positions = normalized_angles * circumference + min_val
    print(positions.shape)

    return positions



def extract_fr_column(spike_data, column):
    column_data = []
    for i in range(len(spike_data)):
        column_data.append(list_of_list_to_1d_numpy_array(spike_data[column].iloc[i]))
    return np.array(column_data)


def compare_ldm(spike_data):
    fr_time_binned = extract_fr_column(spike_data, column="fr_time_binned_smoothed")
    x_time_binned = extract_fr_column(spike_data, column="fr_time_binned_bin_centres")

    # flip axis so its in form (n_samples, n_features)
    fr_time_binned = np.transpose(fr_time_binned)
    x_time_binned = np.transpose(x_time_binned)

    # Assuming the third column of X is the third variable
    third_variable = x_time_binned[:, 0]
    # Standardize the data
    scaler = StandardScaler()
    X_standardized = scaler.fit_transform(fr_time_binned)

    # Initialize PCA, t-SNE, UMAP, and Isomap
    pca = PCA(n_components=20)
    tsne = TSNE(n_components=2, random_state=0, perplexity=5)
    reducer = umap.UMAP(n_components=2)
    isomap = Isomap(n_components=2)

    # Fit and transform the data
    X_pca = pca.fit_transform(X_standardized)
    X_tsne = tsne.fit_transform(X_standardized)
    X_umap = reducer.fit_transform(X_standardized)
    X_isomap = isomap.fit_transform(X_standardized)
    X_pca_then_umap = reducer.fit_transform(X_pca)

    # Define the custom colormap using discrete colors
    colors = ['grey', 'yellow', 'green', 'orange', 'black']
    boundaries = [0, 30, 90, 110, 170, 200]
    custom_cmap = mcolors.ListedColormap(colors)
    norm = mcolors.BoundaryNorm(boundaries, custom_cmap.N, clip=True)

    # Create a figure with 2x2 subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Plot PCA
    scatter_pca = axes[0, 0].scatter(X_pca[:, 0], X_pca[:, 1], c=third_variable, cmap=custom_cmap, norm=norm,alpha=0.05)
    axes[0, 0].set_title('PCA')
    axes[0, 0].set_xlabel('Component 1')
    axes[0, 0].set_ylabel('Component 2')
    axes[0, 0].set_xlim(-3, 2)

    # Plot t-SNE
    scatter_tsne = axes[0, 1].scatter(X_tsne[:, 0], X_tsne[:, 1], c=third_variable, cmap=custom_cmap, norm=norm,alpha=0.05)
    axes[0, 1].set_title('t-SNE')
    axes[0, 1].set_xlabel('Component 1')
    axes[0, 1].set_ylabel('Component 2')

    # Plot UMAP
    scatter_umap = axes[1, 0].scatter(X_umap[:, 0], X_umap[:, 1], c=third_variable, cmap=custom_cmap, norm=norm,alpha=0.05)
    axes[1, 0].set_title('UMAP')
    axes[1, 0].set_xlabel('Component 1')
    axes[1, 0].set_ylabel('Component 2')

    # Plot Isomap
    scatter_isomap = axes[1, 1].scatter(X_isomap[:, 0], X_isomap[:, 1], c=third_variable, cmap=custom_cmap, norm=norm,alpha=0.05)
    axes[1, 1].set_title('Isomap')
    axes[1, 1].set_xlabel('Component 1')
    axes[1, 1].set_ylabel('Component 2')

    # Plot PCA then UMAP
    scatter_pca_then_umap = axes[0,2].scatter(X_pca_then_umap[:, 0], X_pca_then_umap[:, 1], c=third_variable, cmap=custom_cmap, norm=norm,alpha=0.05)
    axes[0,2].set_title('PCA then UMAP')
    axes[0,2].set_xlabel('Component 1')
    axes[0,2].set_ylabel('Component 2')

    # Add a colorbar with alpha=1
    divider = make_axes_locatable(axes[1, 1])
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = plt.colorbar(scatter_pca_then_umap, cax=cax, label='Third Variable')
    cbar.set_alpha(1)
    cbar._draw_all()

    # Adjust the position of the colorbar
    cbar.ax.yaxis.set_ticks_position('left')
    cbar.ax.yaxis.set_label_position('left')

    # Show the plot
    plt.tight_layout()
    plt.savefig("/mnt/datastore/Harry/plot_viewer/compare_pca_umap_tsne_discrete_colorbar.png", dpi=400)
    plt.close()

    # Create a figure with 2x2 subplots
    fig, axes = plt.subplots(2,3, figsize=(18, 12))

    # Plot PCA
    scatter_pca = axes[0, 0].scatter(X_pca[:, 0], X_pca[:, 1], c=third_variable, cmap='twilight_shifted',alpha=0.05)
    axes[0, 0].set_title('PCA')
    axes[0, 0].set_xlabel('Component 1')
    axes[0, 0].set_ylabel('Component 2')
    axes[0, 0].set_xlim(-3, 2)

    # Plot t-SNE
    scatter_tsne = axes[0, 1].scatter(X_tsne[:, 0], X_tsne[:, 1], c=third_variable, cmap='twilight_shifted',alpha=0.05)
    axes[0, 1].set_title('t-SNE')
    axes[0, 1].set_xlabel('Component 1')
    axes[0, 1].set_ylabel('Component 2')

    # Plot UMAP
    scatter_umap = axes[1, 0].scatter(X_umap[:, 0], X_umap[:, 1], c=third_variable, cmap='twilight_shifted',alpha=0.05)
    axes[1, 0].set_title('UMAP')
    axes[1, 0].set_xlabel('Component 1')
    axes[1, 0].set_ylabel('Component 2')

    # Plot Isomap
    scatter_isomap = axes[1, 1].scatter(X_isomap[:, 0], X_isomap[:, 1], c=third_variable, cmap='twilight_shifted',alpha=0.05)
    axes[1, 1].set_title('Isomap')
    axes[1, 1].set_xlabel('Component 1')
    axes[1, 1].set_ylabel('Component 2')

    # Plot PCA then UMAP
    scatter_pca_then_umap = axes[0,2].scatter(X_pca_then_umap[:, 0], X_pca_then_umap[:, 1], c=third_variable, cmap='twilight_shifted', norm=norm,alpha=0.05)
    axes[0,2].set_title('PCA then UMAP')
    axes[0,2].set_xlabel('Component 1')
    axes[0,2].set_ylabel('Component 2')

    # Add a colorbar with alpha=1
    divider = make_axes_locatable(axes[1, 1])
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = plt.colorbar(scatter_pca_then_umap, cax=cax, label='Pos(cm)')
    cbar.set_alpha(1)
    cbar._draw_all()

    # Adjust the position of the colorbar
    cbar.ax.yaxis.set_ticks_position('left')
    cbar.ax.yaxis.set_label_position('left')

    # Show the plot
    plt.tight_layout()
    plt.savefig("/mnt/datastore/Harry/compare_pca_umap_tsne.png", dpi=400)
    plt.close()
    return

def computer_behaviour_kinematics(position_data, xnew_length, xnew_time_bin_size, track_length):
    resampled_behavioural_data = pd.DataFrame()
    trial_numbers = np.array(position_data['trial_number'], dtype=np.int64)
    x_position_cm = np.array(position_data['x_position_cm'], dtype="float64")
    time_seconds = np.array(position_data['time_seconds'], dtype="float64")
    x_position_elapsed_cm = (track_length*(trial_numbers-1))+x_position_cm

    x = time_seconds
    y = x_position_elapsed_cm
    f = interpolate.interp1d(x, y)
    xnew = np.arange(xnew_time_bin_size/2, (xnew_length*xnew_time_bin_size)+xnew_time_bin_size, xnew_time_bin_size)
    xnew = xnew[:xnew_length]
    ynew = f(xnew)
    x_position_cm = ynew%track_length
    speed = np.append(0, np.diff(ynew))
    acceleration = np.append(0, np.diff(speed))
    trial_numbers = (ynew//track_length).astype(np.int64)+1

    resampled_behavioural_data["time_seconds"] = xnew
    resampled_behavioural_data["x_position_cm"] = x_position_cm
    resampled_behavioural_data["speed"] = speed
    resampled_behavioural_data["acceleration"] = acceleration
    resampled_behavioural_data["trial_numbers"] = trial_numbers
    return resampled_behavioural_data

def cebra_test(spike_data, position_data):
    fr_time_binned = extract_fr_column(spike_data, column="fr_time_binned_smoothed")
    x_time_binned = extract_fr_column(spike_data, column="fr_time_binned_bin_centres")

    # flip axis so its in form (n_samples, n_features)
    fr_time_binned = np.transpose(fr_time_binned)
    x_time_binned = np.transpose(x_time_binned)

    behavioural_data = computer_behaviour_kinematics(position_data, xnew_length=len(x_time_binned[:,0]),
                                                     xnew_time_bin_size=settings.time_bin_size, track_length=200)

    x_position_cm = np.array(behavioural_data["x_position_cm"])
    speed = np.array(behavioural_data["speed"]); speed = speed/np.max(speed)
    acceleration = np.array(behavioural_data["acceleration"])
    trial_numbers = np.array(behavioural_data["trial_numbers"])
    time_seconds = np.array(behavioural_data["time_seconds"])
    x_position_cm_xy_cyc = encode_1d_to_2d(positions=x_position_cm)
    x_position_cm_x_cyc = x_position_cm_xy_cyc[:,0]
    x_position_cm_y_cyc = x_position_cm_xy_cyc[:,1]


    max_iterations = 11000
    output_dimension = 3  # here, we set as a variable for hypothesis testing below.
    cebra_model = CEBRA(model_architecture='offset10-model',
                                batch_size=512,
                                learning_rate=3e-4,
                                temperature=10,
                                output_dimension=output_dimension,
                                max_iterations=max_iterations,
                                distance='cosine',
                                conditional='time_delta',
                                device='cuda_if_available',
                                verbose=True,
                                time_offsets=1)

    all_behaviour = np.stack([x_position_cm, x_position_cm_x_cyc, x_position_cm_y_cyc, speed, acceleration, trial_numbers, time_seconds], axis=0)
    all_behaviour = np.transpose(all_behaviour)
    neural_train, neural_test, label_train, label_test = split_data(fr_time_binned, all_behaviour, 0.2)

    cebra_model.fit(neural_train, label_train[:, 1:3])

    cebra_train = cebra_model.transform(neural_train)
    cebra_test = cebra_model.transform(neural_test)

    cebra_pos_decode_train = decoding_pos(cebra_train, cebra_train, label_train[:, 1:3])
    cebra_pos_decode_test = decoding_pos(cebra_train, cebra_test, label_train[:, 1:3])

    original_train = decode_2d_to_1d(label_train[:,1:3])
    original_test = decode_2d_to_1d(label_test[:,1:3])
    decoded_train = decode_2d_to_1d(cebra_pos_decode_train)
    decoded_test = decode_2d_to_1d(cebra_pos_decode_test)

    fig = plt.figure(figsize=(5, 5))
    ax = plt.subplot(111)
    ax.plot(cebra_model.state_dict_['loss'], c='deepskyblue', alpha=0.3, label='position')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel('Iterations')
    ax.set_ylabel('InfoNCE Loss')
    plt.legend(bbox_to_anchor=(0.5, 0.3), frameon=False)
    plt.savefig("/mnt/datastore/Harry/plot_viewer/cebra/hits_cebra_loss_function.png", dpi=400)
    plt.close()


    fig = plt.figure(figsize=(5, 5))
    ax = plt.subplot(111)
    avg_error_by_location, bin_edges = np.histogram(label_test[:, 0], bins=20, range=(0,200), weights=np.abs(original_test-decoded_test))
    avg_error_by_location = avg_error_by_location/np.histogram(label_test[:, 0], bins=20, range=(0,200))[0]
    bin_centres = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    ax.plot(bin_centres, avg_error_by_location)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel('Location')
    ax.set_ylabel('abs error (cm)')
    plt.savefig("/mnt/datastore/Harry/plot_viewer/cebra/decoder_error.png", dpi=400)
    plt.close()

    fig = plt.figure(figsize=(9, 3), dpi=150)
    plt.subplots_adjust(wspace=0.3)
    ax = plt.subplot(111)
    n_samples = 4000
    ax.scatter(label_test[:n_samples, 6], original_test[:n_samples], c='gray', s=1)
    ax.scatter(label_test[:n_samples, 6], decoded_test[:n_samples], c='red', s=1)
    plt.ylabel('Position [cm]')
    plt.xlabel('Time [s]')
    plt.savefig("/mnt/datastore/Harry/plot_viewer/cebra/position_decoding_test.png", dpi=400)
    plt.close()

    fig = plt.figure(figsize=(9, 3), dpi=150)
    plt.subplots_adjust(wspace=0.3)
    ax = plt.subplot(111)
    n_samples = 4000
    ax.scatter(label_train[:n_samples, 6], original_train[:n_samples], c='gray', s=1)
    ax.scatter(label_train[:n_samples, 6], decoded_train[:n_samples], c='red', s=1)
    plt.ylabel('Position [cm]')
    plt.xlabel('Time [s]')
    plt.savefig("/mnt/datastore/Harry/plot_viewer/cebra/position_decoding_train.png", dpi=400)
    plt.close()
    print("")
    print("plotted some decoding")
    print("plotted some cebra stuff")

def split_data(neural_data, label_data, test_ratio):

    split_idx = int(len(neural_data)* (1-test_ratio))
    neural_train = neural_data[:split_idx, :]
    neural_test = neural_data[split_idx:, :]
    label_train = label_data[:split_idx, :]
    label_test = label_data[split_idx:, :]

    return neural_train, neural_test, label_train, label_test



def decoding_pos(emb_train, emb_test, label_train, n_neighbors=36):
    pos_decoder = KNeighborsRegressor(n_neighbors, metric = 'cosine')
    pos_decoder.fit(emb_train, label_train)
    pos_pred = pos_decoder.predict(emb_test)
    return pos_pred


def plot_embeddings(ax, embedding, label, idx_order = (0,1,2), cmap="", viewing_angle=1):
    idx1, idx2, idx3 = idx_order
    if cmap=="track":
        # Define the custom colormap using discrete colors
        colors = ['grey', 'yellow', 'green', 'orange', 'black']
        boundaries = [0, 30, 90, 110, 170, 200]
        custom_cmap = mcolors.ListedColormap(colors)
        norm = mcolors.BoundaryNorm(boundaries, custom_cmap.N, clip=True)
        r=ax.scatter(embedding[:,idx1],embedding[:, idx2], embedding[:, idx3], c=label, cmap=custom_cmap, norm=norm, s=0.5)
    else:
        r=ax.scatter(embedding[:,idx1],embedding[:, idx2], embedding[:, idx3], c=label, cmap=cmap, s=0.5)
    ax.grid(False)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')

    # Remove tick labels
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    # Transparent spines
    ax.w_xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.w_yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.w_zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))

    # Transparent panes
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    # No ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    if viewing_angle == 1:
        ax.view_init(elev=0, azim=0)
    elif viewing_angle == 2:
        ax.view_init(elev=30, azim=45)
    elif viewing_angle == 3:
        ax.view_init(elev=60, azim=30)
    return ax

def main():

    if settings.suppress_warnings:
        warnings.filterwarnings("ignore")

    spike_data = pd.read_pickle("/mnt/datastore/Harry/Cohort11_april2024/derivatives/M21/D16/vr/M21_D16_2024-05-16_14-40-02_VR1/processed/kilosort4/spikes.pkl")
    position_data = pd.read_csv("/mnt/datastore/Harry/Cohort11_april2024/derivatives/M21/D16/vr/M21_D16_2024-05-16_14-40-02_VR1/processed/position_data.csv")

    #compare_ldm(spike_data)

    cebra_test(spike_data, position_data)





if __name__ == '__main__':
    main()