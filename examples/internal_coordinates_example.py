import pyqofta as pqt
import numpy as np
import matplotlib.pyplot as plt

traj_path = '../data/Trajectories/CS2/bound/TRAJ_00001/output.xyz'
traj = pqt.TrajectorySH.init_from_xyz(traj_path)
ICs = traj.calculate_internal_coords()

# example of plotting the cs2 angle over time for a single traj
fig = plt.figure()
plt.plot(traj.time, ICs.angles[:, 0])
plt.show()
print(f'Angle corresponds to connectivity: {ICs.angle_connectivity}')


# same as above except specifying connectivities manually
ang_connect = [[2, 0, 1]]
bond_connect = [[0, 1], [0, 2]]

IC2 = traj.calculate_internal_coords(bond_connect, ang_connect)

