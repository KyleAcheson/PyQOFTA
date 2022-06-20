import pyqofta.trajectories as trj
import pyqofta.molecules as mol
import pyqofta.scattering as scat
import numpy as np
import os
import matplotlib.pyplot as plt

cs2_mol_path = '../data/Molecule/cs2.xyz'
cs2 = mol.Molecule.init_from_xyz(cs2_mol_path)
qmax, Nq = 5, 400
qvec = np.linspace(0, qmax, Nq)

[FF, fq] = scat.IAM_form_factors(cs2, qvec)

plt.plot(qvec, fq)
plt.show()

print('done')






#traj_parent_dir = '../data/Trajectories/CS2/bound/'
#trj_type = 'sh'
#traj_paths = [subdir.path + '/output.xyz' for subdir in os.scandir(traj_parent_dir) if subdir.is_dir()]
#sh_ensemble = trj.Ensemble(traj_paths, trj_type)
