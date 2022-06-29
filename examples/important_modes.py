import numpy as np
import pyqofta as pqt
import matplotlib.pyplot as plt
from natsort import natsorted
from sklearn.cluster import AgglomerativeClustering, FeatureAgglomeration
from scipy.cluster.hierarchy import dendrogram, linkage
import os

##

traj_parent_dir = '/users/kyleacheson/CHD_TRAJS/'
traj_paths = natsorted([traj_parent_dir+fpath for fpath in os.listdir(traj_parent_dir) if 'xyz' in fpath])
chd_ensemble = pqt.Ensemble.load_ensemble(traj_paths, 'sh')
dt = 0.1

def get_distance(traj, connectivity):
    traj_bond_len = np.array(list(traj.broadcast(pqt.Molecule.bond_length, connectivity)))
    return traj_bond_len

def get_dihedral(traj, connectivity):
    traj_dihedral = np.array(list(traj.broadcast(pqt.Molecule.dihedral, connectivity)))
    return traj_dihedral

c1_c6 = [0, 5]
dihedral_a = [1, 2, 3, 4]
dihedral_b = [3, 4, 5, 6]


##

traj = chd_ensemble.trajs[0]
traj_noh = traj.remove_hydrogens()
IC = traj.calculate_internal_coords()

##