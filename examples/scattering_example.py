import pyqofta.trajectories as trj
import pyqofta.molecules as mol
import pyqofta.scattering as scat
import numpy as np
import os
import matplotlib.pyplot as plt

cs2_mol_path = '../data/Molecules/cs2.xyz'
cs2 = mol.Molecule.init_from_xyz(cs2_mol_path)

qmax, Nq = 5, 400 # DEFINE MOMENTUM TRANSFER VECTOR
qvec = np.linspace(0, qmax, Nq)

[FF, fq] = scat.IAM_form_factors(cs2, qvec) # GET FORM FACTORS FOR MOLECULE - ONLY NEED TO DO ONCE
I_xray = scat.IAM_molecular_scattering(cs2, qvec, fq, FF)

fig = plt.figure()
plt.plot(qvec, I_xray) # PLOT OF STATIC CS2 XRS SIGNAL

print('Done single molecule scattering')

traj_parent_dir = '../data/Trajectories/CS2/bound/'
trj_type = 'sh'
traj_paths = [subdir.path + '/output.xyz' for subdir in os.scandir(traj_parent_dir) if subdir.is_dir()] # LIST OF ALL TRAJ PATHS
sh_ensemble = trj.Ensemble(traj_paths, trj_type) # INIT AN ENSEMBLE OBJECT CONTAINING TRAJS ETC.

# FOR AN EXAMPLE WE NOW TAKE ONE TRAJ OUT OF THE ENSEMBLE
trajectory = sh_ensemble.trajs[0]
Itrj = scat.IAM_trajectory_scattering(trajectory, qvec, fq, FF) # SINGLE TRAJ SCATTERING

pdw = np.zeros((trajectory.nts, Nq))
for i in range(trajectory.nts):
    pdw[i, :] = (Itrj[i, :] - Itrj[0, :])/Itrj[0, :] # CALCULATE PERCENTAGE DIFFERENCE SIGNAL (Ion-Ioff / Ioff)

[Q, T] = np.meshgrid(qvec, trajectory.time)
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(Q, T, pdw, edgecolor='none', cmap='RdBu') # SINGLE TRAJECTORY SIGNAL PLOT
plt.xlim([0, 5])

print('Done single trajectory scattering')

# DO ENSEMBLE SCATTERING CALCULATION FOR ALL TRAJS IN sh_ensemble
Iens = scat.IAM_ensemble_scattering(sh_ensemble, qvec, fq, FF) # RETURNS ARRAY W DIMS [sh_ensemble.nts_max, sh_ensemble.ntrajs, Nq]

# HERE WE ADD EACH TRAJS SIGNAL TOGETHER IN AN AVERAGE FASHION
Iavg = np.zeros((sh_ensemble.nts_max, Nq), dtype=float)
for i in range(sh_ensemble.ntrajs):
    Iavg += Iens[:, i, :]
Iavg = Iavg / sh_ensemble.tcount[:, np.newaxis] # ACCOUNT FOR FACT TRAJS MIGHT NOT HAVE EQUAL LENGTH IN TIME

pdw_avg = np.zeros((sh_ensemble.nts_max, Nq), dtype=float)
for i in range(sh_ensemble.nts_max):
    pdw_avg[i, :] = (Iavg[i, :] - Iavg[0, :])/ Iavg[0, :] # PERCENTAGE DIFFERENCE (Ion-Ioff/ Ioff) AGAIN BUT FOR AVERAGE OF ENSEMBLE

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(Q, T, pdw_avg, edgecolor='none', cmap='RdBu') # USING Q AND T AS DEFINED ABOVE - PLOT ENSEMBLE AVERAGE

print('Done ensemble scattering')

# NEED TO ADD CONVOLUTION FUNCTIONS