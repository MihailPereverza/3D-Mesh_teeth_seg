from colormap import hex2rgb
from torch.utils.data import Dataset
import pandas as pd
import torch
import numpy as np
from vedo import *
from scipy.spatial import distance_matrix

import json
import trimesh


class Mesh_Dataset(Dataset):
    def __init__(self, data_list_path, num_classes=15, patch_size=7000, count=None):
        """
        Args:
            h5_path (string): Path to the txt file with h5 files.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data_list = pd.read_csv(data_list_path, header=None)

        # print(self.data_list.shape)
        if count:
            self.data_list = self.data_list.sample(count)
            print(self.data_list)
        # print(self.data_list.shape)
        # print(self.data_list)
        self.num_classes = num_classes
        self.patch_size = patch_size

    def __len__(self):
        return self.data_list.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # i_mesh = self.data_list.iloc[idx][0] #vtk file name
        file_id = self.data_list.iloc[idx, 0]
        mesh_file = f"data/{file_id.replace('_upper', '').replace('_lower', '')}/{file_id}.obj"
        label_file = f"data/{file_id.replace('_upper', '').replace('_lower', '')}/{file_id}.json"

        # Load the mesh using trimesh
        mesh = trimesh.load(mesh_file, process=False)

        fdi_colors = {
            "18": "#ffe2e2", "17": "#ffc6c6", "16": "#ffaaaa", "15": "#ff8d8d", "14": "#ff7171", "13": "#ff5555",
            "12": "#ff3838", "11": "#ff0000", "21": "#0000ff", "22": "#3838ff", "23": "#5555ff", "24": "#7171ff",
            "25": "#8d8dff", "26": "#aaaaff",
            "27": "#c6c6ff",
            "28": "#e2e2ff",
            "38": "#001c00",
            "37": "#003800",
            "36": "#005500",
            "35": "#007100",
            "34": "#008d00",
            "33": "#00aa00",
            "32": "#00c600",
            "31": "#00ff00",
            "48": "#8000ff",
            "47": "#9c38ff",
            "46": "#aa55ff",
            "45": "#b871ff",
            "44": "#c68dff",
            "43": "#d4aaff",
            "42": "#e2c6ff",
            "41": "#f0e2ff",
            "0": "#000000"
        }

        # with open(label_file, 'r') as fp:
        #     json_data = json.load(fp)
        # color label
        # print(len(json_data['labels']))
        # for i, lbl in enumerate(json_data['labels']):
        #     if lbl != 0:
        #         color = list(hex2rgb(fdi_colors[str(lbl)]))
        #         color.append(255)
        #         mesh.visual.vertex_colors[i] = color
        # mesh.export('output.obj')

        points = mesh.vertices
        ids = np.array(mesh.faces)
        # print(f'{ids.shape=}')
        # print(f'{points.shape=}')
        cells = points[ids].reshape(-1, 9).astype(dtype='float32')
        # print(f'{mesh.visual.vertex_colors.shape=}')
        # print(f'{cells.shape=}')
        # print(f'{cells[0]=}')
        # Load labels from the corresponding JSON file
        with open(label_file, 'r') as f:
            labels_data = json.load(f)
        base_labels = labels_data['labels']
        labels_ = np.array(base_labels).astype('int32').reshape(-1, 1)
        # print(f'{labels_.shape=}')
        # print(f'{file_id=}, {labels_.shape=}')
        labels = labels_[ids].reshape(-1, 3)
        # print(f'{labels.shape=}')

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
        # print(f'{X.shape=}, {Y.shape=}')

        # initialize batch of input and label
        X_train = np.zeros([self.patch_size, X.shape[1]], dtype='float32')
        Y_train = np.zeros([self.patch_size, Y.shape[1]], dtype='int32')
        S1 = np.zeros([self.patch_size, self.patch_size], dtype='float32')
        S2 = np.zeros([self.patch_size, self.patch_size], dtype='float32')

        # calculate number of valid cells (tooth instead of gingiva)
        positive_idx = np.argwhere(labels>0)[:, 0] #tooth idx
        negative_idx = np.argwhere(labels==0)[:, 0] # gingiva idx
        # print(f'{negative_idx=}')
        num_positive = len(positive_idx) # number of selected tooth cells

        if num_positive >= self.patch_size * 0.9 // 1: # all positive_idx in this patch
            num_positive = int(self.patch_size * 0.9 // 1)

        num_negative = self.patch_size - num_positive
        positive_selected_idx = np.random.choice(positive_idx, size=num_positive, replace=False)
        negative_selected_idx = np.random.choice(negative_idx, size=num_negative, replace=False)
        selected_idx = np.concatenate((positive_selected_idx, negative_selected_idx))

        selected_idx = np.sort(selected_idx, axis=None)
        # selected_idx = np.random.choice(len(X), size=self.patch_size, replace=False)

        X_train[:] = X[selected_idx, :]
        Y_train[:] = Y[selected_idx, :]
        # print(f'{X_train.shape=}, {Y_train.shape=}')
        if  torch.cuda.is_available():
            TX = torch.as_tensor(X_train[:, :3], device='cuda')
            TD = torch.cdist(TX, TX)
            D = TD.cpu().numpy()
        else:
            D = distance_matrix(X_train[:, :3], X_train[:, :3])

        S1[D<0.1] = 1.0
        S1 = S1 / np.dot(np.sum(S1, axis=1, keepdims=True), np.ones((1, self.patch_size)))

        S2[D<0.2] = 1.0
        S2 = S2 / np.dot(np.sum(S2, axis=1, keepdims=True), np.ones((1, self.patch_size)))

        X_train = X_train.transpose(1, 0)
        Y_train = Y_train.transpose(1, 0)

        sample = {'cells': torch.from_numpy(X_train), 'labels': torch.from_numpy(Y_train),
                  'A_S': torch.from_numpy(S1), 'A_L': torch.from_numpy(S2),
                  # 'file_id': file_id, 'ids': selected_idx,
                  }

        return sample


if __name__ == '__main__':
    dataset = Mesh_Dataset('./train_list.csv')
    dataset.__getitem__(0)
