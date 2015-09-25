import os
import sys

import numpy as np
import scipy as sp
from scipy import ndimage

import util


def readFlow(fn):
    """ Read .flo file in Middlebury format"""
    # Code adapted from:
    # http://stackoverflow.com/questions/28013200/reading-middlebury-flow-files-with-python-bytes-array-numpy

    # WARNING: this will work on little-endian architectures (eg Intel x86) only!
    with open(fn, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        if 202021.25 != magic:
            print 'Magic number incorrect. Invalid .flo file'
            return None
        else:
            w = np.fromfile(f, np.int32, count=1)
            h = np.fromfile(f, np.int32, count=1)
            #print 'Reading %d x %d flo file' % (w, h)
            data = np.fromfile(f, np.float32, count=2*w*h)
            # Reshape data into 3D array (columns, rows, bands)
            return np.resize(data, (h, w, 2))


def flow2hsv(flow, clip=None, clip_pctile=None):
    flowmag = util.norm(flow, axis=2)
    flowang = np.arctan2(util.div_nonz(flow[:,:,1,...],flowmag),
                         util.div_nonz(flow[:,:,0,...],flowmag)) / (2*np.pi) + 0.5
    if clip_pctile is not None:
        clip = np.percentile(flowmag, clip_pctile)
    if clip is None:
        clip = np.amax(flowmag)
    shp = list(flow.shape)
    shp[2] = 1
    extra = np.ones(shp, dtype=flow.dtype)


    flow_pad = np.concatenate((flowang[:,:,np.newaxis,...], extra, flowmag[:,:,np.newaxis,...]/clip), axis=2)
    #flow_pad = np.pad(flow, padw, mode='constant', constant_values=clip)
    return util.hsv2rgb(np.clip(flow_pad,0,1))

""" likely broken
def lk_flow(im1, im2):
    im1 = util.rgb2gray(im1)
    im2 = util.rgb2gray(im2)

    def lk_solve(arg):
        if abs(arg[2]) == 0:
            return [np.nan, np.nan]
        else:
            return np.linalg.solve(arg[0],arg[1])

    dx = sp.ndimage.filters.sobel(im1, axis=1)
    dy = sp.ndimage.filters.sobel(im1, axis=0)
    dt = im2 - im1

    s00 = sp.ndimage.filters.gaussian_filter(dx**2, (1,1))
    s11 = sp.ndimage.filters.gaussian_filter(dy**2, (1,1))
    s01 = sp.ndimage.filters.gaussian_filter(dx*dy, (1,1))

    dtdx = sp.ndimage.filters.gaussian_filter(dt*dx, (1,1))
    dtdy = sp.ndimage.filters.gaussian_filter(dt*dy, (1,1))

    det = s00*s11 - s01**2
    tr = s00 + s11

    flow = np.zeros(im1.shape + (3,))
    flow[...,2] = det - 0.1 * tr**2

    s00 = s00.ravel()  # np.reshape(s00, (-1,))
    s01 = s01.ravel()  # np.reshape(s01, (-1,))
    s11 = s11.ravel()  # np.reshape(s11, (-1,))
    dtdx = dtdx.ravel()
    dtdy = dtdx.ravel()

    det = det.ravel()

    arg = [( [[s00[i],s01[i]],[s01[i],s11[i]]], [dtdx[i],dtdy[i]], det[i] )
           for i in range(len(s00))]
    #f = np.array([lk_solve(arg[i]) for i in range(len(s00))])
    f = map(lk_solve, arg)
    f = np.array(f)

    flow[...,0] = np.reshape(f[:,0], flow.shape[:-1])
    flow[...,1] = np.reshape(f[:,1], flow.shape[:-1])

    return flow
"""