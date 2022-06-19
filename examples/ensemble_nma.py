import pyqofta.trajectories as trj
import pyqofta.molecules as mol
import pyqofta.normal_mode_analysis as nma
import numpy as np
import os

freq_path = '../data/Freq/cs2.molden'
traj_parent_dir = '../data/Trajectories/CS2/bound/'
trj_type = 'sh'
traj_paths = [subdir.path + '/output.xyz' for subdir in os.scandir(traj_parent_dir) if subdir.is_dir()]

sh_ensemble = trj.Ensemble(traj_paths, trj_type)

ref_structure = mol.Vibration(freq_path)
time_intervals = [[0, 2001]]

# call nma analysis on each trajectory in ensemble. Returns map object
nma_trajs = trj.Ensemble.broadcast(nma.nma_traj, sh_ensemble, ref_structure, time_intervals)
nma_trajs = np.array(list(nma_trajs)) # will have shape (ntraj, 2, ntints, nfreqs) - 2 = [avg, std]

[average_normal_modes, nm_std_ensemble] = nma.nm_analysis(sh_ensemble, ref_structure, time_intervals)
