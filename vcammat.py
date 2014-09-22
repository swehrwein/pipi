import numpy as np
import gzip


def saveMat(arr, fn):
    if arr.dtype == np.int32:
        mattype = 'int'
    elif arr.dtype == np.single:
        mattype = 'float'
    elif arr.dtype == np.double:
        mattype = 'double'
    else:
        print "Unsupported array type %s" % str(arr.dtype)
        return

    while np.ndim(arr) < 3:
        arr = np.reshape(arr, arr.shape + (1,))
    with gzip.open(fn, 'wb') as f:
        rows, cols, channels = arr.shape
        f.write('#vcam_mat_v1\n')
        f.write(mattype + '\n')
        f.write('%d %d %d\n' % (cols, rows, channels))

        data = np.zeros(cols*rows*channels)
        for i in range(channels):
            data[i::channels] = arr[:,:,i].ravel()
        f.write(data.tostring())


def loadMat(fn):
    f = gzip.open(fn, 'rb')
    header = f.readline().strip()
    if not header == '#vcam_mat_v1':
        print "vcammat headers do not match (header = %s)" % header
        return np.array([])

    mattype = f.readline().strip()
    if mattype == 'int':
        dt = np.int32
    elif mattype == 'float':
        dt = np.single
    elif mattype == 'double':
        dt = np.double
    else:
        print "vcammat datatype unkonwn: %s" % mattype
        return np.array([])

    (cols, rows, channels) = tuple(int(x) for x in f.readline().split())
    datastr = f.read()
    data = np.fromstring(datastr, dtype=dt)
    f.close()

    data_shaped = np.zeros((rows, cols, channels))
    for i in range(channels):
        data_shaped[:,:,i] = np.reshape(data[i::channels], (rows, cols))
        #print np.reshape(data[i::channels], (rows, cols))

    return data_shaped
    #return np.transpose(np.reshape(data, dims), [1, 0, 2])
