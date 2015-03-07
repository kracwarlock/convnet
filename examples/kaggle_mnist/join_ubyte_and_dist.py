import os, struct
from array import array as pyarray
from numpy import append, array, int8, uint8, zeros
import numpy as np
import h5py as h5

def load_mnist(dataset="training", digits=np.arange(10), path="/ais/gobi3/u/shikhar/mnist/"):
    """
    Loads MNIST files into 3D numpy arrays

    Adapted from: http://abel.ee.ucla.edu/cvxopt/_downloads/mnist.py
    """

    if dataset == "training":
        fname_img = os.path.join(path, 'train-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels.idx1-ubyte')
    elif dataset == "distorted":
        fname_img = os.path.join(path, 'train-images-distorted.idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels.idx1-ubyte')
    else:
        raise ValueError("dataset must be 'testing' or 'training'")

    flbl = open(fname_lbl, 'rb')
    magic_nr, size = struct.unpack(">II", flbl.read(8))
    lbl = pyarray("b", flbl.read())
    flbl.close()

    fimg = open(fname_img, 'rb')
    magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
    img = pyarray("B", fimg.read())
    fimg.close()

    ind = [ k for k in range(size) if lbl[k] in digits ]
    N = len(ind)

    images = zeros((N, rows * cols), dtype=uint8)
    labels = zeros((N, 1), dtype=int8)
    for i in range(len(ind)):
        images[i] = array(img[ ind[i]*rows*cols : (ind[i]+1)*rows*cols ])
        labels[i] = lbl[ind[i]]

    return images, labels

train_data, train_labels = load_mnist()
disto_data, disto_labels = load_mnist(dataset="distorted")

data_ = np.concatenate((train_data,disto_data)).astype(np.float32)/255.0
labels = np.concatenate((train_labels, disto_labels)).astype(int)

f = h5.File('/ais/gobi3/u/shikhar/mnist/distorted_plus_all.h5', 'w')
f.create_dataset('train', data=data_)
f.create_dataset('train_labels', data=labels)
f.close()

