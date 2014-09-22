import numpy as np


def write(fname, coords, normals, colors, mask=None):

    if coords.ndim > 2:
        coords = np.reshape(coords, (-1, 3))

    if mask is None:
        mask = np.ones_like(coords, dtype=bool)
    elif mask.ndim > 2:
        mask = np.reshape(mask, (-1,3))
    elif mask.ndim == 2 and mask.shape[1] != 3:
        mask = np.reshape(mask, (-1,))[:,np.newaxis]

    mask = mask[:,0]

    coords = coords[mask,:]

    N = coords.shape[0]

    fmt = '%f %f %f'

    with open(fname, 'w') as f:
        f.write('ply\n'
                'format ascii 1.0\n'
                'element vertex %d\n'
                'property float x\n'
                'property float y\n'
                'property float z\n' % N)

        data = coords

        if normals is not None:
            if normals.ndim > 2:
                normals = np.reshape(normals, (-1, 3))

            normals = normals[mask,:]

            f.write('property float nx\n'
                    'property float ny\n'
                    'property float nz\n')

            data = np.concatenate((data, normals), axis=1)
            fmt = fmt + ' %f %f %f'

        if colors is not None:
            if colors.ndim == 1:
                colors = colors[:,np.newaxis]
            if colors.ndim > 2:
                colors = np.reshape(colors, (-1, 3))
            if colors.shape[1] != 3:  # handle grayscale case
                colors = np.reshape(colors, (-1,))
                colors = np.column_stack((colors, colors, colors))

            colors = colors[mask,:]

            f.write('property uchar diffuse_red\n'
                    'property uchar diffuse_green\n'
                    'property uchar diffuse_blue\n')

            data = np.concatenate((data, colors), axis=1)
            fmt = fmt + ' %d %d %d'

        fmt = fmt + '\n'
        f.write('end_header\n')

        for i in range(data.shape[0]):
            f.write(fmt % tuple(data[i,:]))
