import pyqofta.trajectories as trj
import pyqofta.molecules as mol
import numpy as np
import os
import matplotlib.pyplot as plt

traj_parent_dir = '../data/Trajectories/CS2/bound/'
trj_type = 'sh'
traj_paths = [subdir.path + '/output.xyz' for subdir in os.scandir(traj_parent_dir) if subdir.is_dir()] # LIST OF ALL TRAJ PATHS
sh_ensemble = trj.Ensemble(traj_paths, trj_type) # INIT AN ENSEMBLE OBJECT CONTAINING TRAJS ETC.

# FOR AN EXAMPLE WE NOW TAKE ONE TRAJ OUT OF THE ENSEMBLE
trajectory = sh_ensemble.trajs[0]

ref_mol = trajectory.geometries[0] # a reference structure chosen by user
rmsd = trajectory.Kabsch_rmsd(ref_mol)

plt.plot(trajectory.time, rmsd)


print('done')