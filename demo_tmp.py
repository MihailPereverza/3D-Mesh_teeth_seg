from torch.utils.data import Dataset
import pandas as pd
import torch
import numpy as np
from vedo import *
from scipy.spatial import distance_matrix

import json
import trimesh

patch_size = 6000

mesh_file = f"data/O52P1SZT/O52P1SZT_lower.obj"
label_file = f"data/O52P1SZT/O52P1SZT_lower.json"

# Load the mesh using trimesh
mesh = trimesh.load(mesh_file, process=False)
points = mesh.vertices
ids = np.array(mesh.faces)
cells = points[ids].reshape(-1, 9).astype(dtype='float32')
# Load labels from the corresponding JSON file
with open(label_file, 'r') as f:
    labels_data = json.load(f)
labels_ = np.array(labels_data['labels']).astype('int32').reshape(-1, 1)
labels = labels_[ids].reshape(-1, 3)

print("cells.shape",cells.shape)
print("labels.shape",labels.shape)

# # new way
# # move mesh to origin
# points = mesh.points()
# mean_cell_centers = mesh.center_of_mass()
# points[:, 0:3] -= mean_cell_centers[0:3]

# Normalize mesh data by moving it to the origin and scaling
# mean_point = points.mean(axis=0)

# std_point = points.std(axis=0)
means = points.mean(axis=0)
stds = points.std(axis=0)
points -= means  # translating all points so the centroid is at the origin

for i in range(3):
    cells[:, i] = (cells[:, i] - means[i]) / stds[i] #point 1
    cells[:, i+3] = (cells[:, i+3] - means[i]) / stds[i] #point 2
    cells[:, i+6] = (cells[:, i+6] - means[i]) / stds[i] #point 3

X = cells
Y = labels

# initialize batch of input and label
X_train = np.zeros([patch_size, X.shape[1]], dtype='float32')
Y_train = np.zeros([patch_size, Y.shape[1]], dtype='int32')
S1 = np.zeros([patch_size, patch_size], dtype='float32')
S2 = np.zeros([patch_size, patch_size], dtype='float32')

# calculate number of valid cells (tooth instead of gingiva)
positive_idx = np.argwhere(labels>0)[:, 0] #tooth idx
negative_idx = np.argwhere(labels==0)[:, 0] # gingiva idx

num_positive = len(positive_idx) # number of selected tooth cells

if num_positive > patch_size: # all positive_idx in this patch
    positive_selected_idx = np.random.choice(positive_idx, size=patch_size, replace=False)
    selected_idx = positive_selected_idx
else:   # patch contains all positive_idx and some negative_idx
    num_negative = patch_size - num_positive # number of selected gingiva cells
    positive_selected_idx = np.random.choice(positive_idx, size=num_positive, replace=False)
    negative_selected_idx = np.random.choice(negative_idx, size=num_negative, replace=False)
    selected_idx = np.concatenate((positive_selected_idx, negative_selected_idx))

selected_idx = np.sort(selected_idx, axis=None)
# selected_idx = np.random.choice(len(X), size=patch_size, replace=False)

X_train[:] = X[selected_idx, :]
Y_train[:] = Y[selected_idx, :]