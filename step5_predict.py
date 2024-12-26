import os
import numpy as np
import torch
import torch.nn as nn
import trimesh
from colormap import hex2rgb

from meshsegnet import *
import vedo
import pandas as pd
from losses_and_metrics_for_mesh import *
from scipy.spatial import distance_matrix
import utils

if __name__ == '__main__':

    gpu_id = utils.get_avail_gpu()

    model_path = os.getcwd() + '/models'
    # model_name = 'Mesh_Segementation_MeshSegNet_50_classes_1500samples_best.tar'
    # model_name = 'Mesh_Segementation_MeshSegNet_50_classes_1500samples_last.tar'
    # model_name = 'Mesh_Segementation_MeshSegNet_50_classes_1500samples_with_zero_best_with_zero.tar'
    # model_name = 'Mesh_Segementation_MeshSegNet_50_classes_1500samples_with_zero_last_with_zero.tar'
    model_name = 'Mesh_Segementation_MeshSegNet_50_classes_1500samples_with_zero_zero_las.tar'
    # model_name = 'Mesh_Segementation_MeshSegNet_50_classes_1500samples_with_zero_zero_best.tar'
    # model_name = 'Mesh_Segementation_MeshSegNet_50_classes_1500samples_with_custom_new_last.tar'
    model_name = 'Mesh_Segementation_MeshSegNet_50_classes_1500samples_for_sasha_new_las.tar'

    mesh_path = './'  # need to define
    teeth_id = 'UNKC1VVC'
    sample_filenames = [f'data/{teeth_id}/{teeth_id}_upper.obj'] # need to define
    output_path = './outputs'
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    num_classes = 50
    num_channels = 9  # number of features

    # set model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MeshSegNet(num_classes=num_classes, num_channels=num_channels).to(device, dtype=torch.float)

    # load trained model
    checkpoint = torch.load(os.path.join(model_path, model_name), map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    del checkpoint
    model = model.to(device, dtype=torch.float)

    #cudnn
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    # Predicting
    model.eval()
    with torch.no_grad():
        for i_sample in sample_filenames:

            print('Predicting Sample filename: {}'.format(i_sample))
            mesh = trimesh.load(os.path.join(mesh_path, i_sample), process=False)

            if mesh.faces.shape[0] > 20000:
                print('\tDownsampling...')

                # Рассчитываем целевое количество ячеек
                target_num = 20000
                ratio = target_num / mesh.faces.shape[0]  # ratio для уменьшения

                # Используем метод decimate (уменьшение числа граней) с заданным коэффициентом
                mesh_d = mesh.simplify_quadric_decimation(1 - ratio)

                # Инициализируем массив меток для уменьшенной сетки
                predicted_labels_d = np.zeros([mesh_d.faces.shape[0], 1], dtype=np.int32)
            else:
                mesh_d = mesh
                predicted_labels_d = np.zeros([mesh_d.faces.shape[0], 1], dtype=np.int32)

            print('\tPredicting...')
            points = mesh_d.vertices
            ids = np.array(mesh_d.faces)
            cells = points[ids].reshape(-1, 9).astype(dtype='float32')

            means = points.mean(axis=0)
            stds = points.std(axis=0)
            points -= means  # translating all points so the centroid is at the origin

            for i in range(3):
                cells[:, i] = (cells[:, i] - means[i]) / stds[i]  # point 1
                cells[:, i + 3] = (cells[:, i + 3] - means[i]) / stds[i]  # point 2
                cells[:, i + 6] = (cells[:, i + 6] - means[i]) / stds[i]  # point 3

            X = cells
            print(X.shape)
            # computing A_S and A_L
            S1 = np.zeros([X.shape[0], X.shape[0]], dtype='float32')
            S2 = np.zeros([X.shape[0], X.shape[0]], dtype='float32')

            if torch.cuda.is_available():
                TX = torch.as_tensor(X[:, :3], device='cuda')
                TD = torch.cdist(TX, TX)
                D = TD.cpu().numpy()
            else:
                D = distance_matrix(X[:, :3], X[:, :3])

            S1[D < 0.1] = 1.0
            S1 = S1 / np.dot(np.sum(S1, axis=1, keepdims=True), np.ones((1, X.shape[0])))

            S2[D < 0.2] = 1.0
            S2 = S2 / np.dot(np.sum(S2, axis=1, keepdims=True), np.ones((1, X.shape[0])))

            X = X.transpose(1, 0)
            X = torch.from_numpy(X).reshape([1, X.shape[0], X.shape[1]]).to(device, dtype=torch.float)
            print(X.size())

            A_S = torch.from_numpy(S1).reshape([1, S1.shape[0], S1.shape[1]]).to(device, dtype=torch.float)
            A_L = torch.from_numpy(S2).reshape([1, S2.shape[0], S2.shape[1]]).to(device, dtype=torch.float)
            print(f'{X.size()=}, {A_S.size()=}, {A_L.size()=}')

            print(X.shape)
            tensor_prob_output = model(X, A_S, A_L).to(device, dtype=torch.float)
            patch_prob_output = tensor_prob_output.cpu().numpy()

            for i_label in range(num_classes):
                predicted_labels_d[np.argmax(patch_prob_output[0, :], axis=-1)==i_label] = i_label

            # output downsampled predicted labels
            mesh2 = mesh_d.copy()
            print(predicted_labels_d)
            # colors from 0 to 50
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

            print(f'{mesh2.faces.shape=}, {mesh2.visual.vertex_colors.shape=}')
            reconstructed_labels = np.zeros(mesh2.vertices.shape[0], dtype='int32')
            reconstructed_labels[mesh2.faces] = predicted_labels_d.flatten().reshape(-1, 1)
            print(np.sum(reconstructed_labels == 0))
            print(f'{reconstructed_labels.shape=}')
            for face_idx, label in enumerate(reconstructed_labels):
                if label == 0:
                    continue

                mesh2.visual.vertex_colors[face_idx] = list(hex2rgb(fdi_colors.get(str(label), "#000000"))) + [255]

            output_file = os.path.join(output_path, '{}_d_predicted.obj'.format(i_sample[:-4].rsplit('/', 1)[-1]))
            mesh2.export(output_file)
            mesh2.show()
