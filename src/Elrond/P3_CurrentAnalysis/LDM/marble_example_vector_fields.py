"""This example illustrates MARBLE for a vector field on a flat surface."""
import numpy as np
import sys
from MARBLE import plotting, preprocessing, dynamics, net, postprocessing
import matplotlib.pyplot as plt
import os
import numpy as np 
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import seaborn as sns
from statannotations.Annotator import Annotator

import cebra

import MARBLE
from P3_CurrentAnalysis.LDM.rat_utils import *

def f0(x):
    return x * 0 + np.array([-1, -1])

def f1(x):
    return x * 0 + np.array([1, 1])

def f2(x):
    eps = 1e-1
    norm = np.sqrt((x[:, [0]] + 1) ** 2 + x[:, [1]] ** 2 + eps)
    u = x[:, [1]] / norm
    v = -(x[:, [0]] + 1) / norm
    return np.hstack([u, v])

def f3(x):
    eps = 1e-1
    norm = np.sqrt((x[:, [0]] - 1) ** 2 + x[:, [1]] ** 2 + eps)
    u = x[:, [1]] / norm
    v = -(x[:, [0]] - 1) / norm
    return np.hstack([u, v])

# def f2(x):
#     eps = 1e-1
#     norm = np.sqrt((x[:, [0]]) ** 2 + x[:, [1]] ** 2 + eps)
#     u = -(x[:, [0]] ) / norm
#     v = -(x[:, [1]] ) / norm
#     return np.hstack([u, v])

# def f3(x):
#     eps = 1e-1
#     norm = np.sqrt((x[:, [0]]) ** 2 + x[:, [1]] ** 2 + eps)
#     u = (x[:, [0]] ) / norm
#     v = (x[:, [1]] ) / norm
#     return np.hstack([u, v])

  

def split_data(data, test_ratio):
    split_idx = int(data['neural'].shape[0] * (1-test_ratio))
    neural_train = data['neural'][:split_idx]
    neural_test = data['neural'][split_idx:]
    label_train = data['continuous_index'][:split_idx]
    label_test = data['continuous_index'][split_idx:]
    return neural_train.numpy(), neural_test.numpy(), label_train.numpy(), label_test.numpy()



def main():
     
    with open('data/rat_data.pkl', 'rb') as handle:
        hippocampus_pos = pickle.load(handle)
        
    hippocampus_pos = hippocampus_pos['achilles']
    neural_train, neural_test, label_train, label_test = split_data(hippocampus_pos, 0.2)
    print("")


    # generate simple vector fields
    # f0: linear, f1: point source, f2: point vortex, f3: saddle
    n = 512
    x = [dynamics.sample_2d(n, [[-1, -1], [1, 1]], "random", seed=i) for i in range(4)]
    y = [f0(x[0]), f1(x[1]), f2(x[2]), f3(x[3])]  # evaluated functions

    # construct data object
    data = preprocessing.construct_dataset(x, y)

    # train model
    model = net(data, params={'inner_product_features': False, 
                              'diffusion': False})
    model.fit(data)

    # evaluate model on data
    data = model.transform(data)
    data = postprocessing.cluster(data)
    data = postprocessing.embed_in_2D(data)

    # plot results
    titles = ["Linear left", "Linear right", "Vortex right", "Vortex left"]
    plotting.fields(data, titles=titles, col=2, width=0.01)
    plt.savefig('fields.png')
    plotting.embedding(data, data.y.numpy(), titles=titles, clusters_visible=True)
    plt.savefig('embedding.png')
    plotting.histograms(data, titles=titles)
    plt.savefig('histogram.png')
    plotting.neighbourhoods(data)
    plt.savefig('neighbourhoods.png')
    plt.show()

     
if __name__ == "__main__":
    main()