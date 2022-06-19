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
    'TrajectorySH',
]


acceptable_traj_type = ['sh', 'mce', 'aimce']

class EnsembleTypeError(TypeError):
    def __init__(self, msg=f'Trajectory types must be one of: {acceptable_traj_type}', *args, **kwargs):
        super().__init__(msg, *args, **kwargs)

class Ensemble:

    def __init__(self, fpaths: list, traj_type: str):
        self.trajs = load_ensemble(fpaths, traj_type)
        self.ntrajs = len(self.trajs)
        self.weights = [1.0/self.ntrajs for i in range(self.ntrajs)]

    def __iter__(self):
        return EnsembleIterator(self)

    @staticmethod
    def load_ensemble(fpaths: list, traj_type: str):
        traj_type = traj_type.lower()
        if traj_type not in acceptable_traj_type:
            raise EnsembleTypeError

        # init lists of trajectory object for the correct dynamics type
        if traj_type == 'sh':
            trajs = [TrajectorySH.init_from_xyz(fpath) for fpath in fpaths]
        elif traj_type == 'mce':
            #trajs = [TrajectoryMCE.init_from_xyz(fpath) for fpath in fpaths]
            raise EnsembleTypeError(f'Trajectory type: {traj_type} is not yet implemented.')
        elif traj_type == 'aimce':
            #trajs = [TrajectoryAIMCE.init_from_xyz(fpath) for fpath in fpaths]
            raise EnsembleTypeError(f'Trajectory type: {traj_type} is not yet implemented.')

        return trajs

    @staticmethod
    def broadcast(func, ensemble, *args):
        map_obj = map(lambda elem: func(elem, *args), ensemble)
        return map_obj




class Trajectory:
    """
    Class to represent a general Trajectory. This is the parent class which is
    inhereted by a series of other classes that represent more specifric trajectory structures.
    For example, trajectories calculated via. Surface Hopping (SH), Multi-configurational Ehrenfest (MCE),
    Ab-Initio Multiple Spawning (AIMS) etc. This class is only used to define methods which are universal
    wrt the type of trajectory.

    Attributes:
    -----------
    self.geometries: list
        list of Molecule objects that describe coorindates and associated properties over all timesteps
    self.time: list
        time vector
    self.nts: int
        number of time steps in trajectory
    self.dt: int
        timestep size (typically fs)
    self.widths: numpy.ndarray
        array of widths for methods with a frozen Gaussian description of the nuclear wavefunction (default = None)
    self.pops: numpy.ndarray
        populations of electronic states (default = None)

    Methods:
    ________
    internal_coordinates

    """

    def __init__(self, geometries, time, widths=None, pops=None):
        self.geometries = geometries
        self.time = time
        self.nts = len(self.time)
        self.dt = self.time[1]-self.time[0]
        self.widths = widths
        self.pops = pops
        self.weight = 1

    def __iter__(self):
        return TrajectoryIterator(self)

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

    @staticmethod
    def broadcast(func, trajectory, *args):
        """
        A method that allows for the broadcasting of a function over elements of an iterable.
        This generalised function may contain additional arguments. This is to allow for the
        broadcast of molecular level functions over the individual timesteps of a trajectory object.
        :param func: any function that takes elements of iterable as an argument
        :param trajectory: an instance of the Trajectory class
        :param args: additional arguments to function
        :return: a Trajectory property with a function applied to its elements
        :rtype: map object
        """
        map_obj  = map(lambda elem: func(elem, *args), trajectory)
        return map_obj



class TrajectorySH(Trajectory):
    """
    Class to represent a Surface Hopping trajectory (may be from Sharc or Newton-X).
    TrajectorySH inherits from the general Trajectory class.
    Here the nuclear wavefunction is a delta-function and so widths are not included as a property.

    Attributes:
    -----------
    self.geometries: list
        list of Molecule objects that describe coorindates and associated properties over all timesteps
    self.time: list
        time vector
    self.nts: int
        number of time steps in trajectory
    self.dt: int
        timestep size (typically fs)

    Methods:
    ________

    Class Methods:
    --------------
    init_from_xyz(cls, fpath)

    Static Methods:
    ---------------
    read_sharc(trj_file)
    """


    def __init__(self, geometries, time):
        self.geometries = geometries
        self.time = time
        self.nts = len(self.time)
        self.dt = self.time[1]-self.time[0]
        self.weight = 1

    @classmethod
    def init_from_xyz(cls, fpath):
        """
        A method to instantiate a trajectory object from an output file
        :param fpath: trajectory output file
        :type fpath:str
        :return: trajectory object
        :rtype: TrajectorySH
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



class TrajectoryIterator:

    def __init__(self, trajectory):
        self._trajectory = trajectory
        self._index = 0


    def __next__(self):
        if self._index < len(self._trajectory.geometries):
            result = self._trajectory.geometries[self._index]
            self._index += 1
            return result
        else:
            raise StopIteration


class EnsembleIterator:

    def __init__(self, ensemble):
        self._ensemble = ensemble
        self._index = 0

    def __next__(self):
        if self._index < len(self._ensemble.trajs):
            result = self._ensemble.trajs[self._index]
            self._index += 1
            return result
        else:
            raise StopIteration


if __name__ == "__main__":
    # testing molecule class
    #molecule_path = 'data/Molecules/cs2.xyz'
    #mol = mol.Molecule.init_from_xyz(molecule_path)
    #D = mol.distance_matrix()
    #mol.gen_internal_coords(D)
    #trajectory_path = 'data/Trajectories/CS2/output.xyz'
    #trj = TrajectorySH.init_from_xyz(trajectory_path)
    #[bonds, angles, dihedrals] = trj.internal_coordinates()

    freq_path = 'data/Freq/freq.molden'
    ref_structure = mol.Vibration(freq_path)
    #norm_mode_matrix = ref_structure.ret_normal_mode_matrix(mass_weight=True)

    trajectory_path = 'data/Trajectories/CS2/output.xyz'
    trj = TrajectorySH.init_from_xyz(trajectory_path)
    trj.norm_mode_transform(ref_structure, mass_weighted=True)
    [avg, std] = trj.nma_analysis([[0, 2001]])

    print('Testing Done')
