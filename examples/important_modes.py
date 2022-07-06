import numpy as np
import pyqofta as pqt
import matplotlib.pyplot as plt
from natsort import natsorted
from sklearn.cluster import AgglomerativeClustering, FeatureAgglomeration
from scipy.cluster.hierarchy import dendrogram, linkage
import os
from dtw import *

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

bond_connect = [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 1]]
angle_connect = [[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6, 1], [6, 1, 2]]
dihedral_connect = [[1, 2, 3, 4], [3, 4, 5, 6], [2, 3, 4, 5], [4, 5, 6, 1], [5, 6, 1, 2], [6, 1, 2, 3]]

bond_connect = np.array(bond_connect)-1
angle_connect = np.array(angle_connect)-1
dihedral_connect = np.array(dihedral_connect)-1

##

traj = chd_ensemble.trajs[1]
traj_noh = traj.remove_hydrogens()


##

def get_internal_coordinates(structure, bond_connect, angle_connect, dihedral_connect):
    nb = np.shape(bond_connect)[0]
    na = np.shape(angle_connect)[0]
    nd = np.shape(dihedral_connect)[0]
    bonds, angles, dihedrals = [], [], []

    for b in range(nb):
        bonds.append(structure.bond_length(list(bond_connect[b, :])))

    for a in range(na):
        angles.append(structure.angle(list(angle_connect[a, :])))

    for d in range(nd):
        dihedrals.append(structure.dihedral(list(dihedral_connect[d, :])))

    return bonds, angles, dihedrals

def traj_internal_coords(traj, bond_connect, angle_connect, dihedral_connect):
    tbonds, tangles, tdihedrals = [], [], []
    for timestep in traj:
        [bonds, angles, dihedrals] = get_internal_coordinates(timestep, bond_connect, angle_connect, dihedral_connect)
        tbonds.append(bonds)
        tangles.append(angles)
        tdihedrals.append(dihedrals)
    return np.array(tbonds), np.array(tangles), np.array(tdihedrals)

def ensemble_internal_coords(ensemble, bond_connect, angle_connect, dihedral_connect):
    ebonds, eangles, edihedrals = [], [], []
    for traj in ensemble:
        traj = traj.remove_hydrogens()
        [bonds, angles, dihedrals] = traj_internal_coords(traj, bond_connect, angle_connect, dihedral_connect)
        #ebonds.append(bonds)
        #eangles.append(angles)
        #edihedrals.append(dihedrals)

        #bond_var = ic_variance(bonds)
        #angle_var = ic_variance(angles)
        #dihedral_var = ic_variance(dihedrals)
        ebonds.append(bonds)
        eangles.append(angles)
        edihedrals.append(dihedrals)
    return np.array(ebonds), np.array(eangles), np.array(edihedrals)


def ic_variance(coord):
    nts, nic = coord.shape
    var = np.zeros(nic)
    for ts in range(nts):
        var += (coord[ts, :] - coord[0, :])**2
    return var/nts

##

#[bonds, angles, dihedrals] = traj_internal_coords(traj_noh, bond_connect, angle_connect, dihedral_connect)

[bonds, angles, dihedrals] = ensemble_internal_coords(chd_ensemble, bond_connect, angle_connect, dihedral_connect)

##

X = np.hstack((bonds[:,-1,:], angles[:,-1,:], dihedrals[:, -1, :]))
X_mean = np.mean(X, axis=0)
X = X - X_mean
Xstd = np.std(X, axis=0)
X = X/ Xstd

U, S, E = np.linalg.svd(X)
plt.plot(np.log(S))

##
from copy import copy
inds = [0, 1, 5, 7, 17, 19, 34, 28, 62]
inds = [0,3,4,5,26,37,65]
inds =[0, 3, 4, 5, 26, 37, 65]
bond_lens = bonds[inds,:,-1]
b = copy(bond_lens)
bond_lens -= np.mean(bond_lens)
bond_lens /= np.std(bond_lens)
dists = []
for i in range(len(inds)):
    if i == 0:
        for j in range(i+1, len(inds)):
            alignment = dtw(bond_lens[i, :], bond_lens[j, :], keep_internals=True)
            print(f'traj{inds[i]} and traj{inds[j]}: {alignment.distance}')
            dists.append(alignment.distance)

fig, ax = plt.subplots()
ax.plot(bond_lens.T)
ax.legend(inds)
plt.show()

##

dtw(bond_lens[0, :], bond_lens[7, :], keep_internals=True,
    step_pattern=rabinerJuangStepPattern(6, "c"))\
    .plot(type="twoway",offset=-2)
plt.show()

##
from sklearn.preprocessing import MinMaxScaler
grid = np.linspace(0, 2*np.pi, 200)
x = np.sin(grid)
y = np.sin(grid+(np.pi/2))

#x -= np.mean(x)
#y -= np.mean(y)
#x /= np.std(x)
#y /= np.std(y)
scaler = MinMaxScaler()

terms = [np.sin(grid+(np.pi/2)), np.sin(grid*(np.pi/2)), 3*np.sin(grid)]
for i in range(len(terms)):
    y = terms[i]
    #y = (y - y.min(axis=0)) / (y.max(axis=0) - y.min(axis=0))
    print(dtw(x, y, keep_internals=True).distance)

##
dtw(x, y, keep_internals=True,
    step_pattern=rabinerJuangStepPattern(6, "c"))\
    .plot(type="twoway",offset=-2)
plt.show()
