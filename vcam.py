import util 

import OpenEXR
import Imath
import numpy

import os


def load_from_rec(dataset, rec): 
    # returns im3d, mask3d, coords2d, im2d
    base_dir = '/home/swehrwein/vcam/data/'
    data_dir = os.path.join(base_dir, dataset, 'data')
    
    im3d_fn = os.path.join(data_dir, 'projpoints', rec['pmvsid']  + '_0000.exr')
    im3d, mask3d, coords2d = imread_projpoints(im3d_fn)
    
    im2d_fn = os.path.join(data_dir, rec['filename'])
    im2d = util.imread(im2d_fn)

    return im3d, mask3d, coords2d, im2d
    

def imread_projpoints(filename):
    # returns img, valid, coords
    exrimage = OpenEXR.InputFile(filename)

    dw = exrimage.header()['dataWindow'] 
    (width, height) = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1) 

    def fromstr(s, datatype): 
        mat = numpy.fromstring(s, dtype=datatype) 
        mat = mat.reshape (height,width) 
        return mat 

    npy_dt = {'HALF':   numpy.float16, 
              'FLOAT':  numpy.float32}

    channels = exrimage.header()['channels']
    
    # image data - RGB or Y
    if all([clr in channels for clr in 'RGB']):
        ch = 'RGB'
    elif 'Y' in channels:
        ch = 'Y'
    im_t = str(channels[ch[0]].type)
    img_data = [fromstr(s, npy_dt[im_t]) for s in exrimage.channels(ch)]
    img = numpy.dstack(img_data)

    # valid mask
    valid = None
    if 'valid' in channels:
        valid_t = str(channels['valid'].type)
        valid = fromstr(exrimage.channel('valid'), npy_dt[valid_t])

    # coordinate data
    coords = None
    coord_ch = ['point_image_coordinate_%s' % c for c in 'xy']
    if all([c in channels for c in coord_ch]):
        coord_t = str(channels[coord_ch[0]].type)
        coord_data = [fromstr(s, npy_dt[coord_t]) 
                      for s in exrimage.channels(coord_ch)]#, exr_dt[coord_t])]
        coords = numpy.dstack(coord_data);

    return img, valid, coords
