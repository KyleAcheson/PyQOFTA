"""
Author: Kyle Acheson
A module containing the basic data structures for analysing quantum dynamics trajectories.
Contains Ensemble, the general Trajectory structure with subclasses for specific
type of trajectories - e.g. Sharc trajectories.

Generally objects are instantiated from external files (xyz, molden etc.).
"""

import numpy as np
import numpy.typing as npt
import pyqofta.molecules as mol


__all__ = [
    'Ensemble',
    'Trajectory',
    'SharcTrajectory',
]





class Ensemble:

    def __init__(self):
        pass



class Trajectory:

    def __init__(self, geometries, time, width=None, pops=None):
        self.geometries = geometries
        self.time = time
        self.nts = len(self.time)
        self.dt = self.time[1]-self.time[0]
        self.width = width
        self.pops = pops
        self.weight = 1

    def rmsd(self, ref_geom):
        pass

    def convolution(self, func):
        pass

    def autocorrolation(self):
        pass

    def internal_coordinates(self):
        pass

    def coarse_grain(self, tsteps):
        pass


class SharcTrajectory(Trajectory):

    trj_file = 'output.xyz'

    def __init__(self, geometries, time):
        self.geometries = geometries
        self.time = time
        self.nts = len(self.time)
        self.dt = self.time[1]-self.time[0]
        self.weight = 1
        self.norm_mode_coords = None

    @classmethod
    def init_from_xyz(cls, fpath):
        """
        A method to instantiate a trajectory object from an output file
        :param fpath: trajectory output file
        :type fpath:str
        :return: trajectory object
        :rtype: SharcTrajectory
        """
        geoms, tvec = cls.read_sharc(fpath)
        return cls(
            geometries=geoms,
            time=tvec
        )

    @staticmethod
    def read_sharc(trj_file):
        """
        A method to read information from SHARC output files (xyz format)
        :param trj_file: output file name
        :type trj_file: str
        :return: geometries over all time steps and time vector
        :rtype: list
        """
        skip_lines = 2 # top two lines of each timestep block have no coordinates
        geometries, coords, labels, time = [], [], [], []
        f = open(trj_file, 'r')
        first_line = f.readline().strip()
        f.close()
        Nat = int(first_line) # number atoms on first line
        with open(trj_file, 'r') as f:
            count = 0;
            for line in f:
                count += 1
                if count == skip_lines:
                    # the line that contains time step info
                    time_list = line.strip().split()
                    time.append(float(time_list[1]))
                elif count > skip_lines and count <= Nat+skip_lines:
                    # we are in the geometry coordinates block
                    atom_list = line.strip().split()
                    coords.append([float(i) for i in atom_list[1:]])
                    labels.append(atom_list[0])
                elif count > Nat+skip_lines:
                    count = 1 # reset to 1 as we are now at natom line
                    geometries.append(mol.Molecule(labels, np.array(coords)))
                    coords, labels = [], []
        return geometries, time

    def internal_coordinates(self):
        """
        A method to generate and return all combinations of internal coordinates over all
        timestep for a given instance of a trajectory.
        :return: seperate arrays for distances, angles and dihedral angles
        :rtype: nump.ndarray
        """
        # THIS IS A BIT JANKY - NEED TO REWRITE AND THNK ABOUT IMPLEMENTATION OF IC
        # OBJECTS AS CURRENTLY IMPLEMENTED.
        distances, angles, dihedrals = [], [], []
        for timestep in self.geometries:
            D = timestep.distance_matrix()
            ICs = timestep.ret_internal_coords(D)
            bond_objs = list(filter(lambda x: type(x) == mol.BondDistance, ICs))
            distances.append([bond.magnitude for bond in bond_objs])
            ang_objs = list(filter(lambda x: type(x) == mol.BondAngle, ICs))
            angles.append([ang.magnitude for ang in ang_objs])
            dih_objs = list(filter(lambda x: type(x) == mol.DihedralAngle, ICs))
            dihedrals.append([dih.magnitude for dih in dih_objs])
        bond_connectivity = [bond.connectivity for bond in bond_objs]
        angle_connectivity = [ang.connectivity for ang in ang_objs]
        dihedral_connectivity = [dih.connectivity for dih in dih_objs]
        return np.array(distances), np.array(angles), np.array(dihedrals)  # init dihedrals as float64 is a waste (if empty)

    def norm_mode_transform(self, ref_structure, mass_weighted=False):
        self.norm_mode_coords = np.zeros((ref_structure.nfreqs, self.nts))
        norm_mode_mat = ref_structure.ret_normal_mode_matrix(mass_weighted)
        for idx, molecule in enumerate(self.geometries):
            displacement_vec = molecule.coordinates.flatten() - ref_structure.coordinates.flatten()
            self.norm_mode_coords[:, idx] = np.dot(displacement_vec, norm_mode_mat)

    def nma_analysis(self, time_intervals: list) -> tuple[npt.NDArray, npt.NDArray]:
        """
        A method for trajectory normal mode analysis. Calculates the average of the normal mode coordinates over
        a number of time intervals specified. Also calculates the standard deviation of these modes within the intervals.
        This gives an indicator of how active a given normal mode is within the trajectory.
        :param time_intervals: a series of time intervals - can just be the whole of time, or can specify a more informed
        selection of intervals that reflect the period of a vibration.
        :type time_intervals: list
        :return: average of the normal mode coordinates and the standard deviation over the set of time intervals
        :rtype: numpy.ndarray
        """
        nvib = np.shape(self.norm_mode_coords)[0]
        ntints = len(time_intervals)
        interval_std = np.zeros((nvib, ntints))
        interval_avg = np.zeros((nvib, ntints))
        for i in range(ntints):
            time_interval = time_intervals[i]
            tstart, tend = time_interval[0], time_interval[1]
            tdiff = tend - tstart
            nm_coords = self.norm_mode_coords[:, tstart:tend] # select normal mode coords over specified time interval
            summed_over_tint = np.sum(nm_coords, axis=1)  # coords summed over time interval (used for std calculation)
            sq_summed_over_tint = np.sum(nm_coords**2, axis=1) # sum of the coords squared over time interval
            if tdiff != 0: # calculate standard dev. of normal modes over each time interval
                avg_tint = summed_over_tint / tdiff
                interval_avg[:, i] = avg_tint
                avg_sq_tint = sq_summed_over_tint / tdiff
                interval_std[:, i] = (tdiff / (tdiff - 1) * (avg_sq_tint - avg_tint ** 2)) ** .5
        return interval_avg, interval_std








if __name__ == "__main__":
    # testing molecule class
    #molecule_path = 'data/Molecules/cs2.xyz'
    #mol = mol.Molecule.init_from_xyz(molecule_path)
    #D = mol.distance_matrix()
    #mol.gen_internal_coords(D)
    #trajectory_path = 'data/Trajectories/CS2/output.xyz'
    #trj = SharcTrajectory.init_from_xyz(trajectory_path)
    #[bonds, angles, dihedrals] = trj.internal_coordinates()

    freq_path = 'data/Freq/freq.molden'
    ref_structure = mol.Vibration(freq_path)
    #norm_mode_matrix = ref_structure.ret_normal_mode_matrix(mass_weight=True)

    trajectory_path = 'data/Trajectories/CS2/output.xyz'
    trj = SharcTrajectory.init_from_xyz(trajectory_path)
    trj.norm_mode_transform(ref_structure, mass_weighted=True)
    [avg, std] = trj.nma_analysis([[0, 2001]])

    print('Testing Done')
