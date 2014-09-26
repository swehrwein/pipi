import os


class cache(object):

    def __init__(self, recompute, filenames, loadfun, savefun):
        self.filenames = filenames
        self.recompute = recompute
        self.loadfun = loadfun
        self.savefun = savefun

    def __call__(self, f):
        def wrapped_f(*args):
            if self.recompute or not all(os.path.exists(fn) for fn in self.filenames):
                results = f(*args)
                for i, r in enumerate(results):
                    self.savefun(r, self.filenames[i])
                    return results
            else:
                return (self.loadfun(fn) for fn in self.filenames)
        return wrapped_f


def cache_or_call(f, args, force_call, filenames, loadfun, savefun):
    if force_call or not all(os.path.exists(fn) for fn in filenames):
        results = f(*args)
        for i, r in enumerate(results):
            savefun(r, filenames[i])
        return results
    else:
        return tuple([loadfun(fn) for fn in filenames])
        #return results[0] if len(results) == 1 else results
        #if len(results) == 1:
        #    return results[0]
        #else:
        #    return results

        #return (loadfun(fn) for fn in filenames)
