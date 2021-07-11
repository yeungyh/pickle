import numpy as np
from collections import namedtuple
import h5py
import time

class GeomMRST(object):

    def __init__(self, filename):
        self.faces = namedtuple('faces', ['num', 'nodes', 'centroids', 'to_hf', 'areas', 'normals', 'neighbors', 'num_interior', 'int_ext'])
        self.cells = namedtuple('cells', ['num', 'nodes', 'centroids', 'to_hf', 'volumes'])
        self.nodes = namedtuple('nodes', ['num', 'coords'])

        with h5py.File(filename, 'r') as f:
            self.faces.num = int(f.get('faces/num')[:].item())
            self.faces.neighbors = f.get('faces/neighbors')[:].astype(int) - 1
            
            is_interior = np.logical_and(*(self.faces.neighbors >= 0))
            self.faces.num_interior = np.count_nonzero(is_interior)
            self.faces.int_ext = np.argsort(~is_interior)
            Ni_range = np.arange(self.faces.num_interior)
            self.faces.to_hf = np.concatenate((Ni_range, Ni_range, np.arange(self.faces.num_interior, self.faces.num)))
            
            self.faces.neighbors = self.faces.neighbors[:, self.faces.int_ext]
            self.faces.nodes = (f.get('faces/nodes')[:].astype(int) - 1).reshape((2, -1), order='F')[:, self.faces.int_ext]
            self.faces.centroids = f.get('faces/centroids')[:][:, self.faces.int_ext]
            self.faces.areas = f.get('faces/areas')[:].ravel()[self.faces.int_ext]
            self.faces.normals = f.get('faces/normals')[:][:, self.faces.int_ext]
            self.faces.normals[:, self.faces.num_interior:] *= np.array([1, -1]).dot(self.faces.neighbors[:, self.faces.num_interior:] >= 0)

            self.cells.num = int(f.get('cells/num')[:].item())
            self.cells.nodes = f.get('cells/nodes')[:].astype(int) - 1
            self.cells.centroids = f.get('cells/centroids')[:]
            self.cells.volumes = f.get('cells/volumes')[:]
            self.cells.to_hf = np.concatenate((self.faces.neighbors[:, :self.faces.num_interior].ravel(),
                                               self.faces.neighbors[:, self.faces.num_interior:].max(axis=0)))

            self.nodes.num = int(f.get('nodes/num')[:].item())
            self.nodes.coords = f.get('nodes/coords')[:]

    def calculate(self):
        pass

    def areas(self):
        polygons = self.nodes.coords.T[self.cells.nodes.T, :]
        return np.abs(np.sum(polygons[:, :, 0] * (np.roll(polygons[:, :, 1], 1, 1) - np.roll(polygons[:, :, 1], -1, 1)), axis=1)) / 2

    def cellsContain(self, points):
        polygons = self.nodes.coords.T[self.cells.nodes.T, :]
        return np.nonzero(np.all(np.cross(polygons - np.roll(polygons, 1, 1), points[:, None, None, :] - polygons[None, :, :, :]) > 0, 2))[1]

    def anyCellsWithin(self, polygons):
        cells = np.nonzero(np.all(np.cross(polygons - np.roll(polygons, 1, 1), self.cells.centroids.T[:, None, None, :] - polygons[None, :, :, :]) > 0, 2))
        cands = cells[0][np.argsort(cells[1])].reshape((-1, 4))
        return cands[np.arange(len(cands)), np.random.random_integers(0, 3, cands.shape[0])]