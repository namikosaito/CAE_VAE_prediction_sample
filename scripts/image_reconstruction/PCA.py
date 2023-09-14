# -*- coding: utf-8 -*-

import numpy as np
from numpy.core.numeric import NaN
import os

import matplotlib.pyplot as plt
import matplotlib.cm as cm

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


ts = 10 #time step

def main():
    ########## file management ##########
    learned_model = "epoch_best"
    file_name = "VAE_20230914_1620"
    save_dir = "../result/" + file_name

    dirs = os.listdir("../result/" + file_name)
    dirs = [s for s in dirs if "epoch" in s]

    ########## PCA setting ##########
    n_components = 8
    plot_axis = [1,2]

    sample_dir = save_dir + "/" + learned_model + "/h_sample"
    print(sample_dir)

    ########## data loader ##########
    sample_files = os.listdir(sample_dir)
    sample_files.sort()
    sequence_length = []
    max_sequence_length = 0
    for i in range(len(sample_files)):
        data = np.load(sample_dir + "/" + sample_files[i])
        sequence_length.append(data.shape[0])
        if data.shape[0] > max_sequence_length:
            max_sequence_length = data.shape[0]
        if i == 0:
            sample_all = data
        else:
            sample_all = np.concatenate([sample_all, data])

    # print(sample_all, sample_all.shape)

    ########## PCA ##########
    pca = PCA(n_components=n_components)
    values = pca.fit_transform(sample_all)
    print(values.shape)

    if not os.path.exists(save_dir + "/" + learned_model + "/pca/pc{}pc{}".format(plot_axis[0], plot_axis[1])):
        os.makedirs(save_dir + "/" + learned_model + "/pca/pc{}pc{}".format(plot_axis[0], plot_axis[1]))

    plt.xlabel("PC{} ({})".format(plot_axis[0], pca.explained_variance_ratio_[plot_axis[0]-1]), fontsize=20)
    plt.ylabel("PC{} ({})".format(plot_axis[1], pca.explained_variance_ratio_[plot_axis[1]-1]), fontsize=20)
    
    plot_axis=[1,2]
    for i in range(len(values)):       
        if i//ts == 0:
            col = "r"
        elif i//ts == 1:
            col = "b"
        else:
            print("black")
        
        if i%ts == 0:
            shape = "*"
        elif i%ts == ts-1:
            shape = "s"
        else:
            shape = "o"
        plt.scatter(values[i][plot_axis[0]-1], values[i][plot_axis[1]-1], color=col, marker = shape, alpha=1.0, s=200)
    plt.savefig(save_dir + "/" + learned_model + "/pca/pc{}pc{}".format(plot_axis[0], plot_axis[1]) + "/train.png")
    plt.show()

if __name__ == "__main__":
    main()