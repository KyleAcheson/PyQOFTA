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

##

c1_c6 = [0, 5]
dihedral_a = [1, 2, 3, 4]
dihedral_b = [3, 4, 5, 6]

distances_map = chd_ensemble.broadcast(get_distance, c1_c6)
distances = np.array(list(distances_map))

dihedral_a_map = chd_ensemble.broadcast(get_dihedral, dihedral_a)
dihedral_b_map = chd_ensemble.broadcast(get_dihedral, dihedral_b)

dihedrals_a = np.array(list(dihedral_a_map))
dihedrals_b = np.array(list(dihedral_b_map))

##

for i in range(99):
    for j in range(75):
        if np.abs(dihedrals_a[i, j+1] - dihedrals_a[i, j]) > 90:
            dihedrals_a[i, j+1] = dihedrals_a[i, j+1] + 360
        if j == 74:
            break


for i in range(99):
    for j in range(75):
        if np.abs(dihedrals_b[i, j+1] - dihedrals_b[i, j]) > 90:
            dihedrals_b[i, j+1] = dihedrals_a[i, j+1] + 360
        if j == 74:
            break



##

# plot c1-c6 distances
plt.rcParams['text.usetex'] = True
plt.plot(np.array(chd_ensemble.trajs[0].time)*dt, np.transpose(distances))
plt.xlabel('$t$ (fs)')
plt.ylabel('$|\mathbf{r}_{\mathrm{C1}} - \mathbf{r}_{\mathrm{C6}}|$ (\AA$^{-1}$)')
plt.autoscale(enable=True, axis='x', tight=True)
plt.show()

##

plt.rcParams['text.usetex'] = True
plt.plot(np.array(chd_ensemble.trajs[0].time)*dt, np.transpose(dihedrals_a))
plt.xlabel('$t$ (fs)')
plt.ylabel('$\phi_{\mathrm{a}} ( \theta)$')
plt.autoscale(enable=True, axis='x', tight=True)
plt.show()

##

plt.rcParams['text.usetex'] = True
plt.plot(np.array(chd_ensemble.trajs[0].time)*dt, np.transpose(dihedrals_b))
plt.xlabel('$t$ (fs)')
plt.ylabel('$\phi_{\mathrm{b}} (\theta)$')
plt.autoscale(enable=True, axis='x', tight=True)
plt.show()


##

dihedrals_end = dihedrals_a[:, -1]
dihedrals_end2 = dihedrals_b[:, -1]
distances_end = distances[:, -1]

##

fig = plt.figure()
plt.rcParams['text.usetex'] = True
ax = fig.add_subplot(111, projection='3d')
ax.scatter(np.abs(dihedrals_end), np.abs(dihedrals_end2), distances_end)
ax.set_xlabel('$|\phi_{\mathrm{a}}|$')
ax.set_ylabel('$|\phi_{\mathrm{b}}$|')
ax.set_zlabel('$|\mathbf{r}_{\mathrm{C1}} - \mathbf{r}_{\mathrm{C6}}|$ (\AA$^{-1}$)')
plt.show()


##
