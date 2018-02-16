import os
import itertools

import util

import cv2
import numpy as np
import scipy as sp


def save_video(filename, video):
    if video.ndim == 3:
        video = util.gray2rgb(video)
    vw = VideoWriter(filename, *(video.shape[:2]))
    vw.write_chunk(video)
    vw.close()


def load_video(filename):
    vr = VideoReader(filename)


def save_frames(filepattern, video):
    for i in range(video.shape[-1]):
        util.imwrite(video[...,i], filepattern % i)


def resize(vid, scale):
    if vid.ndim == 3:
        h, w, f = vid.shape
    else:
        h, w, c, f = vid.shape
    newh = int(h*scale)
    neww = int(w*scale)

    out = np.zeros((newh, neww,) + tuple(vid.shape[2:]), dtype=vid.dtype)
    for i in range(f):
        out[...,i] = cv2.resize(vid[...,i], (neww, newh))
    return out

"""def resize(vid, scale, interp='bilinear'):
    first = util.imresize(vid[...,0], scale)

    out = np.zeros(first.shape + (vid.shape[-1],), dtype=vid.dtype)
    out[...,0] = first
    for i in range(1, vid.shape[-1]):
        out[...,i] = util.imresize(vid[...,i], scale, interp=interp)
    return out"""


"""Classes to wrap MoviePy's pipe-based reader and writers."""

from moviepy.video.io.ffmpeg_reader import FFMPEG_VideoReader
from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter


class VideoReader(FFMPEG_VideoReader):
    def __init__(self, filename, max_frames=None, starttime=0, duration=None, asfloat=True, scale=None):
        FFMPEG_VideoReader.__init__(self, filename, check_duration=True,
                                    starttime=starttime, duration=duration)
        self.width, self.height = self.size
        self.gen = self._gen()
        self.max_frames = max_frames
        self.asfloat = asfloat
        self.scale = scale

    def _gen(self):
        frame = self.read_frame()
        while frame is not None:
            if self.scale is not None:
                frame = util.imresize(frame, self.scale)
            yield (frame.astype(np.float32) / 255.0) if self.asfloat else frame
            frame = self.read_frame()

    def __iter__(self):
        return self

    def next(self):
        #self.current_frame += 1
        #print self.max_frames, self.pos
        if self.max_frames and self.pos >= self.max_frames:
            raise StopIteration
        return next(self.gen)

    def read(self):
        return next(self)

    def read_chunk(self, frames):
        chunk = np.zeros((self.height, self.width, 3, frames), dtype=np.float32)
        i = 0
        for frame in itertools.islice(self,frames):
            #print i
            chunk[...,i] = frame
            i += 1
        chunk = chunk[...,:i]
        return chunk

    def chunk_gen(self, chunk_size):
        chk = self.read_chunk(chunk_size)
        while chk.shape[-1] > 0:
            yield chk
            chk = self.read_chunk(chunk_size)


class VideoWriter(FFMPEG_VideoWriter):
    def __init__(self, filename, height, width, framerate=30,
                 codec="libx264", preset="medium", crf=18, pix_fmt=None):
        params = ["-crf", str(crf)]
        if pix_fmt:
            params.extend(['-pix_fmt', pix_fmt])
        FFMPEG_VideoWriter.__init__(self, filename, (width, height), framerate,
                                    codec=codec, preset=preset,
                                    ffmpeg_params=params)

        self.width = width
        self.height = height
        #self.outfile = av.open(filename, 'w')
        #self.stream = self.outfile.add_stream(codec, framerate)
        #self.stream.height = height
        #self.stream.width = width

    def write(self, frame):
        assert(frame.shape[:2] == (self.height, self.width))
        if frame.dtype != np.uint8:
            frame = (frame*255.0).astype(np.uint8)
        self.write_frame(frame)

    def write_chunk(self, chunk):
        if not chunk.dtype == np.uint8:
            chunk = (chunk * 255.0).astype(np.uint8)
        for i in range(chunk.shape[-1]):
            self.write(chunk[...,i])


def test_videoio(infile, out_dir):
    #base = "/home/swehrwein/gpdata/samford/"
    #infile = base + "raw/150828_08_00.mkv"
    vr = VideoReader(infile, max_frames=200)
    #import ipdb; ipdb.set_trace()  # XXX BREAKPOINT

    height, width = vr.height, vr.width
    chunksize = 1024

    import numpy as np
    import pipi
    #chunk = np.zeros((height, width, 3, chunksize),dtype=np.float32)
    chunk = vr.read_chunk(1024)

    #with pipi.Timer("ff read..."):
        #for i, frame in enumerate(itertools.islice(vr,chunksize)):
            #chunk[...,i] = frame

    print chunk.shape, chunk.min(), chunk.max()

    vw = VideoWriter(os.path.join(out_dir, "ff_ov.mkv"), height, width, codec='libx264', crf=0)

    with pipi.Timer("ff write..."):
        vw.write_chunk(chunk)
            #pipi.imwrite(frame, base + "results/ff_ov_%d.png" % i)
    vw.close()


def test_moviepy(infile, out_dir):
    import numpy as np
    import pipi
    from moviepy.video.io.ffmpeg_reader import FFMPEG_VideoReader
    from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter
    chunksize = 128

    vr = FFMPEG_VideoReader(infile)
    w,h = vr.size

    chunk = np.zeros((h, w, vr.depth, chunksize), dtype=np.uint8)
    with pipi.Timer("mp read..."):
        for i in range(chunksize):

            frame = vr.read_frame()
            chunk[...,i] = frame

    vw = FFMPEG_VideoWriter(os.path.join(out_dir, "mp_ov.mkv"), (w,h), 30, codec="libx264", preset="fast", ffmpeg_params=["-crf", "0"])

    with pipi.Timer("mp write..."):
        for i in range(chunksize):
            vw.write_frame(chunk[...,i])

    vr.close()
    vw.close()
