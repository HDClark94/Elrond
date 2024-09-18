import pandas as pd
import Elrond.settings as settings

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
import cebra

from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
import sklearn.metrics

from matplotlib.collections import LineCollection
from ...Helpers.array_utility import list_of_list_to_1d_numpy_array

import sklearn.metrics
import numpy as np
import sys
from MARBLE import plotting, preprocessing, dynamics, net, postprocessing
import matplotlib.pyplot as plt
import torch
import MARBLE
# Define decoding function with kNN decoder. For a simple demo, we will use the fixed number of neighbors 36.
def decoding_pos_dir(embedding_train, embedding_test, label_train, label_test):
    pos_decoder = cebra.KNNDecoder(n_neighbors=36, metric="cosine")

    pos_decoder.fit(embedding_train, label_train)
    pos_pred = pos_decoder.predict(embedding_test)

    test_score = sklearn.metrics.r2_score(label_test, pos_pred)
    pos_test_err = np.median(abs(pos_pred - label_test))
    pos_test_score = sklearn.metrics.r2_score(label_test, pos_pred)

    prediction_error = abs(pos_pred - label_test)

    return test_score, pos_test_err, pos_test_score, pos_pred, prediction_error



def split_data(data, test_ratio):

    split_idx = int(data['neural'].shape[0] * (1-test_ratio))
    neural_train = data['neural'][:split_idx]
    neural_test = data['neural'][split_idx:]
    label_train = data['continuous_index'][:split_idx]
    label_test = data['continuous_index'][split_idx:]
    
    return neural_train.numpy(), neural_test.numpy(), label_train.numpy(), label_test.numpy()



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


def computer_behaviour_kinematics(position_data, xnew_length, xnew_time_bin_size, track_length):
    resampled_behavioural_data = pd.DataFrame()
    trial_numbers = np.array(position_data['trial_number'], dtype=np.int64)
    x_position_cm = np.array(position_data['x_position_cm'], dtype="float64")
    time_seconds = np.array(position_data['time_seconds'], dtype="float64")
    x_position_elapsed_cm = (track_length*(trial_numbers-1))+x_position_cm

    x = time_seconds
    y = x_position_elapsed_cm
    f = interpolate.interp1d(x, y)
    xnew = np.arange(xnew_time_bin_size/2, (xnew_length*xnew_time_bin_size)+
                     xnew_time_bin_size, xnew_time_bin_size)
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


def build_Marble_input(rates,labels,pca=None,pca_n=10,delta=1.5):
    if pca is None:
        pca =  PCA(n_components=pca_n)
        rates_pca = pca.fit_transform(rates.T)
    else:
        rates_pca = pca.transform(rates.T)
        
    vel_rates_pca = np.diff(rates_pca, axis=0)

    rates_pca = rates_pca[:-1,:] # skip last

    labels = labels[:rates_pca.shape[0]]
     
    data = MARBLE.construct_dataset(
        anchor=rates_pca,
        vector=vel_rates_pca,
        k=100,
        delta=delta, 
    )
    return data, labels, pca
 


spike_data = pd.read_pickle("/mnt/datastore/Harry/Cohort11_april2024/derivatives/M21/D26/vr/M21_D26_2024-05-28_17-04-41_VR1/processed/kilosort4/spikes.pkl")
position_data = pd.read_csv("/mnt/datastore/Harry/Cohort11_april2024/derivatives/M21/D26/vr/M21_D26_2024-05-28_17-04-41_VR1/processed/position_data.csv")


fr_time_binned = extract_fr_column(spike_data, column="fr_time_binned_smoothed")
x_time_binned = extract_fr_column(spike_data, column="fr_time_binned_bin_centres")

# flip axis so its in form (n_samples, n_features)
fr_time_binned = np.transpose(fr_time_binned)
x_time_binned = np.transpose(x_time_binned)

behavioural_data = computer_behaviour_kinematics(position_data, xnew_length=len(x_time_binned[:,0]),
                                                    xnew_time_bin_size=settings.time_bin_size, 
                                                    track_length=200)
x_position_cm = np.array(behavioural_data["x_position_cm"])
speed = np.array(behavioural_data["speed"]); speed = speed/np.max(speed)
acceleration = np.array(behavioural_data["acceleration"])
trial_numbers = np.array(behavioural_data["trial_numbers"], dtype=np.int64)
time_seconds = np.array(behavioural_data["time_seconds"])
x_position_cm_xy_cyc = encode_1d_to_2d(positions=x_position_cm)
x_position_cm_x_cyc = x_position_cm_xy_cyc[:,0]
x_position_cm_y_cyc = x_position_cm_xy_cyc[:,1]

continuous_behaviours = np.stack([x_position_cm, 
                                    x_position_cm_x_cyc, 
                                    x_position_cm_y_cyc, 
                                    speed, acceleration, 
                                    time_seconds], axis=0).T

continuous_behaviours = np.stack([x_position_cm_x_cyc, 
                                    x_position_cm_y_cyc], axis=0).T

n_samples = 5000 
dataset = {'neural': torch.Tensor(fr_time_binned[:n_samples]), 
            'continuous_index': torch.Tensor(continuous_behaviours[:n_samples]),
            'discrete_index': torch.Tensor(trial_numbers[:n_samples])}  
dataset = {'neural': torch.Tensor(fr_time_binned[:n_samples]), 
            'continuous_index': torch.Tensor(continuous_behaviours[:n_samples])} 
     
neural_train, neural_test, label_train, label_test = split_data(dataset, 0.2) 
print("now building marble data")

data_train, label_train_marble, pca = build_Marble_input(neural_train.T, label_train, pca_n=40)
data_test, label_test_marble, _ = build_Marble_input(neural_test.T, label_test, pca=pca)
print("now training marble model")

# build model
params = {
    "epochs": 100,
    "order": 1,  # order of derivatives
    "hidden_channels": [64],  # number of internal dimensions in MLP
    "out_channels": 32, 
    "inner_product_features": False,
    "emb_norm": True, # spherical output embedding
    "diffusion": False,
    "include_positions": True,
    }

model = MARBLE.net(data_train, params=params) #define model
model.fit(data_train, outdir=f"outputs/marble") # train model
print("") 

print("I have finished fitting the MARBLE model")



