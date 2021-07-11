import numpy as np
import h5py

class BCMRST(object):

    def __init__(self, geom, filename):
        num_ext_faces = geom.faces.num - geom.faces.num_interior
        self.kind = np.empty(num_ext_faces, dtype='<U1')
        self.init_val = np.zeros(num_ext_faces)
        
        ext = geom.faces.int_ext[geom.faces.num_interior:]
        ext_idx = np.argsort(ext)
        sorted_ext = ext[ext_idx]
        with h5py.File(filename, 'r') as f:
            idx = ext_idx[np.searchsorted(sorted_ext, f.get('index')[:].ravel() - 1)]
            self.init_val[idx] = f.get('value')[:]
            self.kind[idx] = np.vectorize(chr)(f.get('kind')[:])
        self.val = np.copy(self.init_val)
        self.mean = {k : np.mean(self.init_val[self.kind == k]) for k in np.unique(self.kind)}

    def randomize(self, kind, scale, rs=np.random.RandomState()):
        self.val[self.kind == kind] = rs.normal(self.init_val[self.kind == kind], self.mean[kind] * scale)
    
    def increment(self, kind, value):
        self.val[self.kind == kind] += value
        
    def rescale(self, kind, scale):
        self.val[self.kind == kind]      /= np.exp(scale)
        self.init_val[self.kind == kind] /= np.exp(scale)
        self.mean[kind]                  /= np.exp(scale)