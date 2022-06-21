import pyqofta.trajectories as trj
import pyqofta.molecules as mol
import pyqofta.scattering as scat
import numpy as np
import os
import matplotlib.pyplot as plt

cs2_mol_path = '../data/Molecules/cs2.xyz'
cs2 = mol.Molecule.init_from_xyz(cs2_mol_path)
qmax, Nq = 5, 400
qvec = np.linspace(0, qmax, Nq)

[FF, fq] = scat.IAM_form_factors(cs2, qvec)
I_xray = scat.IAM_molecular_scattering(cs2, qvec, fq, FF)

[FF, fq] = scat.IAM_form_factors(cs2, qvec, ELEC=True)
I_elec = scat.IAM_molecular_scattering(cs2, qvec, fq, FF, Elec=True)
fig = plt.figure()
plt.plt(qvec, I_xray, qvec, I_elec)


print('Done single molecule scattering')

traj_parent_dir = '../data/Trajectories/CS2/bound/'
trj_type = 'sh'
traj_paths = [subdir.path + '/output.xyz' for subdir in os.scandir(traj_parent_dir) if subdir.is_dir()]
sh_ensemble = trj.Ensemble(traj_paths, trj_type)


trajectory = sh_ensemble.trajs[0]
Itrj = scat.IAM_trajectory_scattering(trajectory, qvec, fq, FF)

pdw = np.zeros((trajectory.nts, Nq))
for i in range(trajectory.nts):
    pdw[i, :] = (Itrj[i, :] - Itrj[0, :])/Itrj[0, :] # percentage difference : Ion-Ioff/ Ioff

[Q, T] = np.meshgrid(qvec, trajectory.time)
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(Q, T, pdw, edgecolor='none', cmap='RdBu')
plt.xlim([0, 5])

print('Done single trajectory scattering')

Iens = list(scat.IAM_ensemble_scattering(sh_ensemble, qvec, fq, FF))

Ien = list(Iens)

# DO SOME PLOTTING OF EACH TRAJECTORY - OR AVERAGE OVER LIST

print('Done ensemble scattering')
