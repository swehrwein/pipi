import util

import os
import cv2
import numpy as np
import scipy as sp


def open_read(fn):
    """Open cv2.VideoCapture.
        open_read(filename)
    """
    vr = cv2.VideoCapture(fn)
    assert(vr.isOpened())
    return vr


def open_write(fn, h, w):
    """Open cv2.VideoWriter.
        open_write(filename, height, width)
        Uses HFYU codec, 30fps
    """
    #fourcc = cv2.VideoWriter_fourcc(*"HFYU")
    fourcc = cv2.cv.CV_FOURCC(*"HFYU")
    # TODO automate level numbering
    outvid = cv2.VideoWriter(fn, fourcc, 30, (w,h))
    assert(outvid.isOpened())
    return outvid


def get_size(vr):
    h = int(vr.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
    w = int(vr.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
    c = 3  # let's just assume this for now, shall we?
    return h, w, c


def read_chunk(videoreader, nframes, scale=None):
    h, w, c = get_size(videoreader)
    #h,w = videoreader.height, videoreader.width
    chunk = None
    for i in range(nframes):
        isframe, frame = videoreader.read()

        if not isframe:
            if chunk is None:
                return None
            else:
                chunk = chunk[...,:i]
                break

        if scale is not None:
            frame = sp.misc.imresize(frame, scale)

        if chunk is None:
            chunk = np.zeros(frame.shape + (nframes,), dtype=np.uint8)

        chunk[...,i] = frame
    return chunk.astype(np.float32)[...,::-1,:] / 255.0


def write_chunk(videowriter, chunk):
    for i in range(chunk.shape[-1]):
        videowriter.write(chunk[...,::-1,i])


def save_video(filename, video):
    vw = open_write(filename, *(video.shape[:2]))
    write_chunk(vw, video)
    vw.release()


def save_frames(filepattern, video):
    for i in range(video.shape[-1]):
        util.imwrite(video[...,i], filepattern % i)



def resize(vid, scale):
    first = sp.misc.imresize(vid[...,0], scale)

    out = np.zeros(first.shape + (vid.shape[-1],))
    out[...,0] = first
    for i in range(1, vid.shape[-1]):
        out[...,i] = sp.misc.imresize(vid[...,i], scale)
    return out

""" Good, except for a bug that makes the right few pixels go nuts."""
import av


class VideoReader(object):
    def __init__(self, filename):
        self.container = av.open(filename)
        self.stream = next(s for s in self.container.streams if s.type == b'video')
        self.height = self.stream.height
        self.width = self.stream.width
        self.gen = self._gen()

    def read(self):
        return next(self)

    def __next__(self):
        return next(self)

    def next(self):
        try:
            return next(self.gen)
        except StopIteration:
            return None

    def _gen(self):
        for packet in self.container.demux(self.stream):
            for frame in packet.decode():
                yield frame.to_rgb().to_nd_array()

    def close(self):
        self.container.close()


class VideoWriter(object):
    def __init__(self, filename, height, width, codec='ffv1', framerate=30):
        self.outfile = av.open(filename, 'w')
        self.stream = self.outfile.add_stream(codec, framerate)
        self.stream.height = height
        self.stream.width = width

    def write(self, frame):
        assert(frame.shape[:2] == (self.stream.height, self.stream.width))
        avframe = av.VideoFrame.from_ndarray(np.ascontiguousarray(frame))
        packet = self.stream.encode(avframe)
        if packet:
            self.outfile.mux(packet)

    def close(self):
        self.outfile.close()


def test(infile, out_dir):
    #base = "/home/swehrwein/gpdata/samford/"
    #infile = base + "raw/150828_08_00.mkv"
    vr = VideoReader(infile)
    #import ipdb; ipdb.set_trace()  # XXX BREAKPOINT

    height, width = vr.stream.height, vr.stream.width
    chunksize = 1024

    import numpy as np
    import pipi
    chunk = np.zeros((height, width, 3, chunksize),dtype=np.uint8)

    with pipi.Timer("ff read..."):
        for i in range(chunksize):
            chunk[...,i] = vr.read()

    print chunk.shape, chunk.min(), chunk.max()

    vw = VideoWriter(os.path.join(out_dir, "ff_ov.avi"), height, width)

    with pipi.Timer("ff write..."):
        for i in range(chunksize):
            frame = chunk[...,i]

            #vw.write((frame*255).astype(np.uint8))
            vw.write(frame)
            #pipi.imwrite(frame, base + "results/ff_ov_%d.png" % i)
    vw.close()

    # test opencv
    vr = open_read(infile)
    with pipi.Timer("cv read..."):
        chunk = read_chunk(vr, chunksize)
    with pipi.Timer("cv write..."):
        save_video(os.path.join(out_dir, "cv_ov.mkv"), (chunk * 255).astype(np.uint8))
    #save_frames(base + "results/cv_ov_%d.png", chunk)

