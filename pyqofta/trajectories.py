"""
Author: Kyle Acheson
A module containing the basic data structures for analysing quantum dynamics trajectories.
Contains `Ensemble`, the general `Trajectory` structure with subclasses for specific
type of trajectories - e.g. Sharc trajectories (`TrajectorySH`). The idea is that one can easily
construct these structures from the molecular building blocks in the `molecules.py` module.
Trajectories are defined as a group of `Molecule` type objects and Ensembles as a group of `Trajectory` type objects.
These can be generated easily by providing a parent directory path that contains all trajecoties as subdirectories.

In the future, one may add additional subclasses of `Trajectory` such as TrajectoryMCE which will define properties
that are unique to MCE trajectories (such as widths and populations over all states). By building the code in this
fashion, we can build external modules such as `scattering.py` which have a series of similar functions for each use case,
these functions can be defined to work only on the correct type of trajectory. For example, the treatment of scattering
between SH and MCE can be different if one accounts for the overlap of the MCE basis functions. The ease of quickly
defining an external function for a specific use case within the framework laid out here is in the fact that one
can easily *broadcast* these operations over the `Ensemble` and `Trajectory` types.

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
    """
    A class to describe an ensemble of trajectories. This is instantiated by providing a list of file paths to
    every trajectory you would like to read in, along with the argument for the trajectory type.
    For example, if `fpaths` is a list of the paths to N SH trajectory files - `Ensemble(fpaths, 'sh')`.
    By default the ensemble is configured with weights equal to `1/Ntraj`, this can be explioted in averaging
    over trajectories, or in the future an external observable fitting module will allow for the optimisation
    of these weights to experimental data.

    The Ensemble object is defined as being iterable, so that if one does:
    ```
    ensemble = Ensemble(fpaths, 'sh')
    for traj in ensemble:
        # do some stuff
    ```
    you can iterate over the list of individual `Trajectory` objects defined within the property `self.trajs`.

    The `broadcast` method allows one to load an external module easily and broadcast functions over the whole
    series of trajectories contained within the ensemble. This will return a map generator object which
    can be converted to a list easily. For example,

    ```
    ensemble.broadcast(some_function, arg1, arg2, ...)
    ```

    will broadcast the trajectory level function:

    ```
    def some_function(trajectory, arg1, arg2, ...):
        return something
    ```
    over all trajectory elements of the list `ensemble.trajs`.

    Attributes
    ----------

    trajs: list
        a list of individual `Trajectory` objects instantiated from file
    ntrajs: int
        number of trajectories in total
    nts_max: int
        maximum number of timesteps over all trajectories (trajs may not all run for same time period)
    tcount: numpy.ndarray
        an vector of length `nts_max` that has the number of trajectories at each time step (used for averaging)
    weights: list
        a list of floats containing trajectory weights (default is equal weighting)

    """

    def __init__(self, fpaths: list, traj_type: str):
        self.trajs, self.nts_max, self.tcount = self.load_ensemble(fpaths, traj_type)
        self.ntrajs = len(self.trajs)
        self.weights = [1.0/self.ntrajs for i in range(self.ntrajs)]


    def __iter__(self):
        return EnsembleIterator(self)

    @staticmethod
    def load_ensemble(fpaths: list, traj_type: str) -> tuple[list, int, npt.NDArray]:
        """
        A method to load an ensemble of trajectories

        Parameters
        ----------

        fpaths : list
            a list of all trajectory files to iterate over and load
        traj_type : str
            type of trajectory - acceptable values are `sh, mce, aimce` (note sh only implemented currently).
            This will determine what subclass of `Trajectory` is instantiated.

        Returns
        -------
        trajs: list
            a list of trajectory objects
        max_time: int
            maximum time the trajectories run for
        tcount: numpy.ndarray
            a vector of length max_time with the number of trajectories present at that given time step
            (used for averaging trajectories that have an inconsistent number of time steps)
        """
        traj_type = traj_type.lower()
        if traj_type not in acceptable_traj_type:
            raise EnsembleTypeError

        # init lists of trajectory object for the correct dynamics type
        max_time, trajs = 0, []
        for fpath in fpaths:
            if traj_type == 'sh':
                traj = TrajectorySH.init_from_xyz(fpath)
                trajs.append(traj)
                if traj.nts > max_time: max_time = traj.nts
            else:
                raise EnsembleTypeError(f'Trajectory type: {traj_type} is not yet implemented.')

        tcount = np.zeros(max_time, dtype=int)
        for traj in trajs:
            tcount[:traj.nts] += 1

        return trajs, max_time, tcount


    def average_(self):
        averaged_ensemble = np.zeros((self.trajs[0].natoms, self.nts_max))
        for trj in self:
            for ts, mol in enumerate(trj):
                averaged_ensemble[:, ts] += mol.coordinates
        averaged_ensemble /= self.tcount
        return averaged_ensemble


    def broadcast(self, func, *args):
        """
        Broadcast a general function over the instance of `Ensemble`

        Parameters
        ----------
        func : function
            a general function that operates on `Trajectory` types
        args :
            a list of all arguments that the function depends on apart from the `Trajectory` instance itself

        Returns
        -------
        map_obj: map object
            a map object that yields the result of the function applied to each trajectory in `self.trajs`
        """
        map_obj = map(lambda elem: func(elem, *args), self)
        return map_obj

    #TODO: DEFINE A GENERALISED FILTER LIKE METHOD




class Trajectory:
    """
    Class to represent a general Trajectory. This is the parent class which is
    inhereted by a series of other classes that represent more specific trajectory structures.
    For example, trajectories calculated via. Surface Hopping (SH), Multi-configurational Ehrenfest (MCE),
    Ab-Initio Multiple Spawning (AIMS) etc. This class is only used to define properties and methods which are universal
    wrt the type of trajectory, for example the `broadcast` and `internal coordinate` methods.
    *Instances of Trajectory are not instantiated themselves*, instead one instantiates `TrajectorySH` or `TrajectoryMCE` etc.
    objects which both inherent shared properties and methods from the Trajectory class here.

    The properties `pops` and `widths` are by default set to None type.

    In a similar fashion to the `Ensemble` class, Trajectory objects are also iterable. So one can do:

    ```
    trajectory = TrajectorySH.init_from_xyz(fpath) # here we are using a SH trajectory as an example
    for timestep in trajectory:
        # do some stuff to molecule objects that make up trajectory
    ```
    this iterates over all the molecule objects within the trajectory.

    Attributes
    ----------

    geometries: list
        list of Molecule objects that describe coorindates and associated properties over all timesteps
    time: list
        time vector
    nts: int
        number of time steps in trajectory
    dt: int
        timestep size (typically fs)
    widths: numpy.ndarray
        array of widths for methods with a frozen Gaussian description of the nuclear wavefunction (default = None)
    pops: numpy.ndarray
        populations of electronic states (default = None)

    """

    def __init__(self, geometries, time, widths=None, pops=None):
        self.geometries = geometries
        self.time = time
        self.nts = len(self.time)
        self.dt = self.time[1]-self.time[0]
        self.widths = widths
        self.pops = pops

    def __iter__(self):
        return TrajectoryIterator(self)

    def calculate_internal_coords(self):
        """
        A method to calculate *all* internal coordinates over the whole trajectory.
        If the user is only interested in a selection of internal coordinates, for example one bond length or angle etc.,
        it is recommended to call the `bond_length`, `angle` or `dihedral` methods directly by specifying the atom
        connectivity.

        Returns
        -------
        internal_coords: InternalCoordinates
            a set of internal coordinates described by an InternalCoordinates object. This contains
            an attribute for each IC and the list of connectivities for each. In the case that of di- and tri-atomic
            molecules that have no defined angle or dihedral, those instance variables will be empty.
            In the case of a trajectory, the bond/ angles/ dihedral properties of `InternalCoordinates`
            are 2D numpy.ndarrays with dimensions [trajectory.nts, number bond lenghts/ angles/ dihedrals]
        """
        bond_lengths, angles, dihedrals = [], [], []
        for idx, timestep in enumerate(self):
            ICs = timestep.calculate_internal_coords() # ret an InternalCoordinates type
            bond_lengths.append(ICs.bonds)
            angles.append(ICs.angles)
            dihedrals.append(ICs.dihedrals)
            if idx == 0:
                bond_connect = ICs.bond_connectivity
                angle_connect = ICs.angle_connectivity
                dihedral_connect = ICs.dihedral_connectivity
        return mol.InternalCoordinates(bond_lengths,
                                       bond_connect,
                                       angles,
                                       angle_connect,
                                       dihedrals,
                                       dihedral_connect)


    def broadcast(self, func, *args):
        """
        Broadcast a general function over the instance of `Trajectory` - applies function to the `Molecule`
        objects that make up the trajectories.

        Parameters
        ----------
        func : function
            a general function that operates on `Molecule` types
        args :
            a list of all arguments that the function depends on apart from the `Molecule` instance itself

        Returns
        -------
        map_obj: map object
            a map object that yields the result of the function applied to each trajectory in `self.trajs`
        """
        map_obj  = map(lambda elem: func(elem, *args), self)
        return map_obj



class TrajectorySH(Trajectory):
    """
    Class to represent a Surface Hopping trajectory (may be from Sharc or Newton-X).
    TrajectorySH inherits from the general Trajectory class.
    Here the nuclear wavefunction is a delta-function and so widths are not included as a property.

    Attributes
    -----------
    geometries: list
        list of Molecule objects that describe coorindates and associated properties over all timesteps
    time: list
        time vector
    nts: int
        number of time steps in trajectory
    dt: int
        timestep size (typically fs)

    """


    def __init__(self, geometries, time):
        self.geometries = geometries
        self.time = time
        self.nts = len(self.time)
        self.dt = self.time[1]-self.time[0]
        self.weight = 1

    @classmethod
    def init_from_xyz(cls, fpath: list):
        """
        A method to instantiate a SH trajectory object from an xyz file

        Parameters
        ----------
        fpath : str
            path to trajectory output file

        """
        geoms, tvec = cls.read_sharc(fpath)
        return cls(
            geometries=geoms,
            time=tvec
        )

    @staticmethod
    def read_sharc(trj_file: str):
        """
        A method to read information on trajectories from SHARC output files (xyz files)

        Parameters
        ----------
        trj_file : str
            file path

        Returns
        -------
        geometries: list
            list of geometries (cartesian coordinates) over all time steps

        time: list
            time vector

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
            geometries.append(mol.Molecule(labels, np.array(coords))) # quick patch for a bug where reader misses last geom
        return geometries, time


    def Kabsch_rmsd(self, reference_structure):
        """
        Calculates the molecule RMSD via the Kabsch algorithm for a whole trajectory wrt some reference_structure.
        This reference structure is typically chosen to be an important representative structure that corresponds to
        a conical intersection, some minima or the time-zero structure.

        Parameters
        ----------
        reference_structure : Molecule
            a structure to take the RMSD wrt to

        Returns
        -------
        trj_rmsd: numpy.ndarray
            a vector containing the RMSD over time
        """
        trj_rmsd = self.broadcast(mol.Molecule.Kabsch_rmsd, reference_structure)
        return np.array(list(trj_rmsd))



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
    # some testing stuff
    print('done')