import numpy as np
import h5py


def saveMat(array, filename, name="data", compression="gzip"):
    f = h5py.File(filename, 'w')
    f.create_dataset(name, data=array, compression=compression)
    f.close()


def loadMat(filename):
    with h5py.File(filename, 'r') as f:
        return np.array(f[f.keys()[0]])
