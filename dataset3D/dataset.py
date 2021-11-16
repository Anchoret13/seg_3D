from os import listdir
from os.path import isfile, join
import glob
import os

import numpy as np
import open3d as o3d
from torch.utils.data import Dataset

import MinkowskiEngine as ME
'''
class ExampleDataset(Dataset):
    def __init__(self):
        self.train_dir = '/home/dyf/dataset/custom_ycb/0020/labelPly'
        self.train_filenames = glob.glob(self.train_dir + "/*")
        self.train_filename = self.train_filenames[0]
        self.quantization_size = 0.01

        print("Training dataset size:", len(self.train_filenames))
        for file_path in self.train_filenames:
            print(file_path)


    def __len__(self):
        return len(self.train_filenames)

    def __getitem__(self, i):
              
        pcd = o3d.io.read_point_cloud(self.train_filename).voxel_down_sample(voxel_size=0.02)
        # pcd = pcd.voxel_down_sample(voxel_size=0.02)
        xyz = np.asarray(pcd.points)
        rgb = np.asarray(pcd.colors)

        input = xyz
        feats = input 
        labels = rgb

        # Quantize the input
        discrete_coords, unique_feats, unique_labels = ME.utils.sparse_quantize(
            coordinates=input,
            features=feats,
            labels=labels,
            quantization_size=self.quantization_size,
            ignore_label=-100)

        return discrete_coords, unique_feats, unique_labels
'''

class ExampleDataset(Dataset):
    def __init__(self, video:str):
        '''
        self.train_dir = '/home/dyf/dataset/custom_ycb/0020/trainPly/'
        self.train_filenames = glob.glob(self.train_dir + "/*")
        self.label_dir = '/home/dyf/dataset/custom_ycb/0020/labelPly/'
        self.label_filenames = glob.glob(self.label_dir + "/*")
        #self.train_filename = self.train_filenames[0]
        self.quantization_size = 0.01
        print("Training dataset size:", len(self.train_filenames))
        for file_path in self.train_filenames:
            print(file_path)
        '''
        self.video = video
        self.video_folder = os.path.join('/home/dyf/dataset/custom_ycb/', video)
        self.train_dir = os.path.join(self.video_folder,'trainPly/')
        self.train_filenames = glob.glob(self.train_dir + "/*")
        self.label_dir = os.path.join(self.video_folder,'labelPly/')
        self.label_filenames = glob.glob(self.label_dir + "/*")
        #self.train_filename = self.train_filenames[0]
        self.quantization_size = 0.01

    def __len__(self):
        return len(self.train_filenames)

    def __getitem__(self, i):
              
        train_pcd = o3d.io.read_point_cloud(self.train_filenames[i]).voxel_down_sample(voxel_size=0.02)
        #print("TRAIN_DATA", self.train_filenames[i])
        label_pcd = o3d.io.read_point_cloud(self.label_filenames[i]).voxel_down_sample(voxel_size=0.02)
        #print("LABEL_DATA", self.label_filenames[i])
        # pcd = pcd.voxel_down_sample(voxel_size=0.02)
        #xyz = np.asarray(train_pcd.points)
        #rgb = np.asarray(train_pcd.colors)
        xyz = np.asarray(label_pcd.points)
        label = np.asarray(label_pcd.colors)


        input = xyz
        feats = input
        labels = label

        # Quantize the input
        discrete_coords, unique_feats, unique_labels = ME.utils.sparse_quantize(
            coordinates=input,
            features=feats,
            labels=labels,
            quantization_size=self.quantization_size,
            ignore_label=-100)

        return discrete_coords, unique_feats, unique_labels