import pyqofta as pqt
import numpy as np
import os
from natsort import natsorted
import matplotlib.pyplot as plt

traj_parent_dir = '/users/kyleacheson/CHD_TRAJS/'
traj_paths = natsorted([traj_parent_dir+fpath for fpath in os.listdir(traj_parent_dir)]) # list and sort all files numerically
ensemble = pqt.Ensemble(traj_paths, 'sh')


def get_distance(traj, connectivity):
    traj_bond_len = np.array(list(traj.broadcast(pqt.Molecule.bond_length, connectivity)))
    return traj_bond_len

c1_c6 = [0, 5]
distances_map = ensemble.broadcast(get_distance, c1_c6)
distances = np.array(list(distances_map))

dt = 0.1

plt.rcParams['text.usetex'] = True
plt.plot(np.array(ensemble.trajs[0].time)*dt, np.transpose(distances))
plt.xlabel('$t$ (fs)')
plt.ylabel('$|\mathbf{r}_{\mathrm{C1}} - \mathbf{r}_{\mathrm{C6}}|$ (\AA$^{-1}$)')
plt.autoscale(enable=True, axis='x', tight=True)

print('done')