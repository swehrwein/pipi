import os
import sys

import numpy as np
import scipy as sp
from scipy import ndimage

import util
from video import VideoReader, VideoWriter


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

        dims = np.fromfile(f, np.int32, count=2)
        w, h = dims[0], dims[1]
        #print 'Reading %d x %d flo file' % (w, h)
        data = np.fromfile(f, np.float32, count=2*w*h)
        # Reshape data into 3D array (columns, rows, bands)
        return np.reshape(data, (h, w, 2))


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


class FlowVideoWriter(VideoWriter):
    """ lossy! """
    def __init__(self, filename, height, width, flow_max):
        VideoWriter.__init__(self, filename, height, width, codec="libx264rgb", crf=0, pix_fmt="bgr24")
        self.flow_max = flow_max
        np.savetxt(filename + ".np", np.array([self.flow_max]))

    def write(self, frame):
        quantframe = np.round(((frame / self.flow_max / 2) + 0.5)*255).astype(np.uint8)
        flowframe = np.concatenate((quantframe, np.zeros_like(quantframe[:,:,0][:,:,np.newaxis])), axis=2)
        VideoWriter.write(self, flowframe)

    def write_chunk(self, chunk):
        for i in range(chunk.shape[-1]):
            self.write(chunk[...,i])


class FlowVisWriter(VideoWriter):
    def __init__(self, filename, height, width, flow_max):
        VideoWriter.__init__(self, filename, height, width)
        self.flow_max = flow_max

    def write(self, frame):
        frame_vis = flow2hsv(frame, clip=self.flow_max)

        VideoWriter.write(self, frame_vis)

    def write_chunk(self, chunk):
        for i in range(chunk.shape[-1]):
            self.write(chunk[...,i])


class FlowVideoReader(VideoReader):
    def __init__(self, filename, max_frames=None):
        VideoReader.__init__(self, filename, max_frames)
        self.flow_max = float(np.loadtxt(filename + ".np"))

    def _gen(self):
        frame = self.read_frame()
        while frame is not None:
            yield ((frame[:,:,:2].astype(np.float32) / 255.0) - 0.5) * 2 * self.flow_max
            frame = self.read_frame()


def save_flow_video(filename, flow_vid, max_flow=5.0):
    fvw = FlowVideoWriter(filename, flow_vid.shape[0], flow_vid.shape[1], max_flow)
    fvw.write_chunk(flow_vid)
    fvw.close()


def save_flow_vis(filename, flow_vid, max_flow=5.0):
    fvw = FlowVisWriter(filename, flow_vid.shape[0], flow_vid.shape[1], max_flow)
    fvw.write_chunk(flow_vid)
    fvw.close()


def flow2vis(flow_vid, flow_vis):
    fvr = FlowVideoReader(flow_vid)
    fvw = FlowVisWriter(flow_vis, fvr.height, fvr.width, fvr.flow_max)
    for frame in fvr:
        fvw.write(frame)
    fvw.close()


def flo2vid(flo_pattern, flovid_name):
    import glob
    flowfiles = sorted(glob.glob(flo_pattern))
    f1 = readFlow(flowfiles[0])
    fvw = FlowVideoWriter(flovid_name, f1.shape[0], f1.shape[1], 5.0)
    for f in flowfiles:
        fvw.write(readFlow(f))
    fvw.close()


def test_flowio():

    h, w, f = 720, 480, 100
    flowfield = np.zeros((h,w,3,f))
    for i in range(100):
        flowfield[4*i:4*i+10,4*i:4*i+10,0,i] = 10.0 + i/20.0
        flowfield[4*i:4*i+10,4*i:4*i+10,1,i] = 5.0 + i/10.0

    fvw = FlowVideoWriter("/home/swehrwein/gpdata/flowvidtest.mkv", h, w, flowfield.max())
    for i in range(f):
        fvw.write(flowfield[...,:2,i])
    fvw.close()

    fvr = FlowVideoReader("/home/swehrwein/gpdata/flowvidtest.mkv")
    myflowfield = np.zeros((h,w,3,f))
    for i in range(f):
        myflowfield[...,:2,i] = fvr.next()

    diff = myflowfield - flowfield
    print np.max(diff), np.max(flowfield)
    vis = np.concatenate((flowfield, myflowfield), axis=1)
    vis = vis / np.max(vis)
    util.implay(vis)


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
