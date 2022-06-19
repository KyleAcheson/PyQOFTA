import pyqofta.trajectories as trj
import pyqofta.molecules as mol
import pyqofta.normal_mode_analysis as nma
import numpy as np


# An example of how to project a single trajectory into normal mode coordinates
# and calculate the activity of the normal modes within several seperate
# time intervals in that trajectory


# instantiate a Vibration object containing referance normal modes and freq info
freq_path = '../data/Freq/cs2.molden'
ref_structure = mol.Vibration(freq_path)

# instantiate a Surface Hopping Trajectory object
trajectory_path = '../data/Trajectories/CS2/bound/TRAJ_00001/output.xyz'
cs2_traj = trj.TrajectorySH.init_from_xyz(trajectory_path)

# using the normal mode analysis module broadcast the normal mode transform over
# all timesteps of a trajectory wrt the reference normal modes in ref_structure
norm_mode_coords = trj.Trajectory.broadcast(nma.normal_mode_transform, cs2_traj, ref_structure) # returns a map object (not allocated to memory yet)
norm_mode_coords = np.array(list(norm_mode_coords)) # to get as np.array if you need to allocate it

tints = [[0, 1000], [1000, 2001]]
[avg, std] = nma.nm_analysis(cs2_traj, ref_structure, tints) # higher std = more active
