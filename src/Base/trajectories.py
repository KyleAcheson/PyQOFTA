"""
Author: Kyle Acheson
A module containing the basic data structures for analysing quantum dynamics trajectories.

If an atom required is not parameterised, you may add it to the dictionary: periodic_table.

Generally objects are instantiated from external files (xyz, molden etc.).
Objects of type Molecule and Atom can be instantiated without an external file, but
for the instantiation of an object of type Vibration one must provide frequency information
from an external molden file.
"""

import numpy as np
import numpy.typing as npt

periodic_table = {
    'H': {'Z': 1.008, 'nelec': 1},
    'C': {'Z': 12.011, 'nelec': 6},
    'N': {'Z': 14.007, 'nelec': 7},
    'O': {'Z': 15.999, 'nelec': 8},
    'S': {'Z': 32.06, 'nelec': 16},
    'Br': {'Z': 79.904, 'nelec': 35},
    'I': {'Z': 126.90, 'nelec': 53}
}


#class Ensemble:
#
#    def __init__(self):
#        pass



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
                    geometries.append(Molecule(labels, np.array(coords)))
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
            bond_objs = list(filter(lambda x: type(x) == BondDistance, ICs))
            distances.append([bond.magnitude for bond in bond_objs])
            ang_objs = list(filter(lambda x: type(x) == BondAngle, ICs))
            angles.append([ang.magnitude for ang in ang_objs])
            dih_objs = list(filter(lambda x: type(x) == DihedralAngle, ICs))
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




class Molecule(object):
    """
    Class to represent a molecule

    Attributes
    ----------
    atom_labels : list
        atomic symbols for each atom in molecule
    coordinates : numpy.ndarray
        molecule coordinates
    natoms : list
        total number of atoms in molecule
    nelecs : list
        number of electrons in each individual atom in molecule
    Zs : list
        mass of each individual atom in molecule
    momenta : numpy.ndarray
        momentum vectors for each atom in molecule

    Methods
    _______
    init_from_xyz(cls, fpath: str)
    read_xyz_mol(fname: str)
    init_from_molden(cls, fpath: str)
    read_molden(fname: str)
    """

    def __init__(self, labels, coordinates, momenta=None):
        """
        Constructor for Molecule object
        :param labels: labels of each atom in molecule
        :type labels: list
        :param coordinates: molecular coordinates
        :type coordinates: numpy.ndarray
        :param momenta: momentum vectors for each atom (optional)
        :type momenta: numpy.ndarray
        """
        self._atom_labels = self.__check_atom_labels(labels)
        self.natoms, self.nelecs, self.Zs = self.__get_atoms_info()
        self._coordinates = coordinates
        self._momenta = momenta
        self.internal_coords = None #generated by gen_internal_coords method
        #try:
        #    if (freqs.all() and vibs.all()) is not None:
        #        self.__check_freqs(freqs, vibs)
        #except AttributeError:
        #    pass
        #self._freqs = freqs
        #self._vibs = vibs


    @property
    def coordinates(self):
        return self._coordinates

    @coordinates.setter
    def coordinates(self, coords: npt.NDArray):
        self.__check_array_coords(self.natoms, coords, 'Molecular coordinates')
        self._coordinates = coords

    @property
    def atom_labels(self):
        return self._atom_labels

    @staticmethod
    def __check_atom_labels(labels: list):
        """
        Checks atomic labels are correct type and upper case
        """
        if len(labels) != len(list(filter(str, labels))):
            raise Exception("Atomic labels must be a list of strings for each element")
        labels = [label.upper() for label in labels]
        return labels

    def __get_atoms_info(self) -> tuple[int, list, list]:
        """
        A private method to get atomic information
        :return number atoms, number electrons per atom, mass of each atom:
        :rtype: int, list, list
        """
        nelec_list, Z_list = [], []
        natoms = len(self._atom_labels)
        for atom in self._atom_labels:
            Z_list.append(periodic_table[atom]['Z'])
            nelec_list.append(periodic_table[atom]['nelec'])
        return (natoms, nelec_list, Z_list)

    @property
    def momenta(self):
        return self._momenta

    @momenta.setter
    def momenta(self, momentum_vecs: npt.NDArray):
        self.__check_array_coords(self.natoms, momentum_vecs, 'Momentum')
        self._momenta = momentum_vecs

    @staticmethod
    def __check_array_coords(natoms: int, array: npt.NDArray, property: str):
        """
        Private method to check that property is defined for all molecular coordinates
        and has the correct type and dimensionality
        :param natoms: number of atoms in molecule
        :type natoms: int
        :param array: An array that defines a property for some coordinates
        :type array: numpy.ndarray
        :param property: Name of the property to check
        :type property: str
        """
        if type(array) != np.ndarray:
            raise Exception("%s must be specified as a numpy array" % property)
        dims = np.shape(array)
        if len(dims) != 2 and dims[1] != 3:
            raise Exception("%s must be an array with dimensions (natom, 3)" % property)
        if dims[0] != natoms:
            raise Exception("%s must be defined for %d atoms." % (property, natoms))

    @classmethod
    def init_from_xyz(cls, fpath: str):
        """
        A class method to initialise a Molecule object from an xyz file
        :param fpath: file path to xyz file
        :type fpath: str
        :return: Molecule object
        :rtype: Molecule
        """
        atom_names, coords = cls.read_xyz_mol(fpath)
        return cls(
            labels=atom_names,
            coordinates=coords
        )

    @staticmethod
    def read_xyz_mol(fname: str) -> tuple[list, npt.NDArray]:
        """
        A method to read a single set of molecular coordinate from an xyz file
        :param fname: name of xyz file
        :type fname: str
        :return: atomic labels and molecular cartesian coordinates
        :rtype: list and numpy.ndarray
        """
        skip_lines = 1
        ftype = fname.split('.')[-1]
        if ftype != 'xyz':
            raise Exception("File is not xyz format")
        atom_coords, labels = [], []
        with open(fname, 'r') as f:
            first_line = f.readline().strip()
            Nat = int(first_line)
            for idx, line in enumerate(f):
                if idx >= skip_lines and idx <= Nat+skip_lines:
                    atom_list = line.strip().split()
                    atom_coords.append([float(i) for i in atom_list[1:]])
                    labels.append(atom_list[0])
        atom_coords = np.array(atom_coords)
        return labels, atom_coords

    @classmethod
    def init_from_molden(cls, fpath: str):
        """
        A class method to initiate a Molecule type from a molden file
        :param fpath: path to molden file
        :type fpath: str
        :return: Molecule object
        :rtype: Molecule
        """
        atom_labels, coords = cls.read_molden(fpath)
        return cls(
            labels=atom_labels,
            coordinates=coords
        )

    @staticmethod
    def read_molden(fname: str) -> tuple[list, npt.NDArray]:
        """
        A method to read ONLY coordinate information from molden files
        :param fname: name of molden file
        :type fname: str
        :return: atomic labels and molecular cartesian coordinates
        :rtype: list, numpy.ndarray
        """
        ftype = fname.split('.')[-1]
        if ftype != 'molden':
            raise Exception("File type must be molden")

        labels, atoms = [], []
        mfile = open(fname, 'r')
        Atoms = False
        for line in mfile:
            if '[' in line or '--' in line:
                Atoms = False
            if '[Atoms]' in line:
                Atoms = True
            elif Atoms:
                words = line.split()
                labels += words[0]
                atom_vec = words[3:6]
                atoms += [[eval(coords) for coords in atom_vec]]

        atoms = np.array(atoms)
        return labels, atoms

    def distance_matrix(self) -> npt.NDArray:
        """
        Method to calculate distance matrix from already instantiated molecule object
        :return: symmetric matrix of atomic distances
        :rtype: numpy.ndarray
        """
        return self.__compute_distance_matrix(self.natoms, self.coordinates)

    #@classmethod
    #def distance_matrix(cls, coords: npt.NDArray):
    #    """
    #    Method to calculate and return distance matrix from an external set of molecular coordinates
    #    :param coords: Molecular coordinates
    #    :type coords: numpy.ndarray
    #    :return: symmetric matrix of atomic distances
    #    :rtype: numpy.ndarray
    #    """
    #    natom = np.shape(coords)[0]
    #    cls.__check_array_coords(natom, coords, 'Molecular coordinates')
    #    return cls.__compute_distance_matrix(natom, coords)


    @staticmethod
    def __compute_distance_matrix(natoms: int, coords: npt.NDArray) -> npt.NDArray:
        """
        Private method that computes the distance matrix for a set of coordinates.
        Cannot be called outside of the class - called internally within the instance and class method wrappers
        :param natoms: number of atoms
        :type natoms: int
        :param coords: molecular coordinates
        :type coords: numpy.ndarray
        :return: symmetric distance matrix
        :rtype: numpy.ndarray
        """
        dist_mat = np.zeros((natoms, natoms))
        for i in range(natoms):
            for j in range(i+1, natoms):
                rvec = coords[i, :] - coords[j, :]
                dist_mat[i, j] = dist_mat[j, i] = np.linalg.norm(rvec)
        return dist_mat

    def angle(self, angle_connectivity: list):
        """
        Caclulates angle between two bond lengths r_ij and r_kj where atom j is central
        :param angle_connectivity: Connectivity of atoms (i, j, k) that form an angle
        :type angle_connectivity: list
        :return: angle between atoms r_ij and r_kj in degrees
        :rtype: float
        """
        i, j, k = angle_connectivity
        r_ij = self.coordinates[i, :] - self.coordinates[j, :]
        r_kj = self.coordinates[k, :] - self.coordinates[j, :]
        cosine_theta = np.dot(r_ij, r_kj)
        sin_theta = np.linalg.norm(np.cross(r_ij, r_kj))
        theta = np.arctan2(sin_theta, cosine_theta)
        theta = 180.0 * theta / np.pi
        return theta

    def dihedral(self, dihedral_connectivity):
        """
        Method to calculate a dihedral angle between two bond lengths that form a plane
        :param dihedral_connectivity: Connectivity of atoms that form a dihedral angle (i, j, k, l)
        :type dihedral_connectivity: list
        :return: dihedral angle (phi) in degrees
        :rtype: float
        """
        i, j, k, l = dihedral_connectivity
        r_ji = self.coordinates[j, :] - self.coordinates[i, :]
        r_kj = self.coordinates[k, :] - self.coordinates[j, :]
        r_lk = self.coordinates[l, :] - self.coordinates[k, :]
        v1 = np.cross(r_ji, r_kj)
        v1 /= np.linalg.norm(v1)
        v2 = np.cross(r_lk, r_kj)
        v2 /= np.linalg.norm(v2)
        p1 = np.cross(v1, r_kj) / np.linalg.norm(r_kj)
        a = np.dot(v1, v2)
        b = np.dot(p1, v2)
        phi = np.arctan2(b, a)
        phi = -180.0 - 180.0 * phi / np.pi
        if phi < -180.0:
            phi += 360.0
        return phi

    def gen_internal_coords(self, distance_mat: npt.NDArray):
        self.internal_coords = self.ret_internal_coords(distance_mat)

    def ret_internal_coords(self, distance_mat: npt.NDArray):
        """
        A method that takes a distance matrix and calculates all possible internal coordinates
        (bond distances, angles, and dihedrals) as well as their connectivity.
        :param distance_mat: symmetric distance matrix of a molecule
        :type distance_mat: numpy.ndarray
        """
        int_coords = []
        if self.natoms > 1:
            atom = self.atom_labels[1]
            r = distance_mat[0][1]
            bond_connectivity = [0, 1]
            int_coords.append(BondDistance(bond_connectivity, r))
        if self.natoms > 2:
            atom = self.atom_labels[2]
            r = distance_mat[0][2]
            bond_connectivity = [0, 2]
            int_coords.append(BondDistance(bond_connectivity, r))
            ang_connectivity = [2, 0, 1]
            ang = self.angle(ang_connectivity)
            int_coords.append(BondAngle(ang_connectivity, ang))
        if self.natoms > 3:
            for i in range(3, self.natoms):
                atom = self.atom_labels[i]
                r = distance_mat[i - 3][i]
                bond_connectivity = [i - 3, 1]
                int_coords.append(BondDistance(bond_connectivity, r))
                ang_connectivity = [i, i - 3, i - 2]
                ang = self.angle(ang_connectivity)
                int_coords.append(BondAngle(ang_connectivity, ang))
                dihedral_connectivity = [i, i - 3, i - 2, i - 1]
                dih = self.dihedral(dihedral_connectivity)
                int_coords.append(DihedralAngle(dihedral_connectivity, dih))
        return int_coords


    def write_xyz(self, fpath):
        pass

    def write_zmat(self, fpath):
        pass

class Vibration(Molecule):
    """
    Class to represent vibrational modes of a molecule structure - subclass of Molecule

    Attributes
    ----------
    self.freqs : list
        vibrational frequencies for each of the 3N modes (in cm^-1)
    self.modes : nump.ndarray
        vectors of each of the normal mode frequencies

    + All attributes inherited from Molecule class

    Methods
    _______
    init_from_xyz(cls, fpath: str)
    read_xyz_mol(fname: str)
    init_from_molden(cls, fpath: str)
    read_molden(fname: str)
    """

    def __init__(self, fpath: str):
        """
        Constructor for Vibration object - loads vibrational information from molden format
        :param fpath: path to molden file
        :type fpath: str
        """
        # init via molden reader - returns atoms, labels and freqs - init Molecule type and additional Vib instances
        molden_out = self.read_molden(fpath)
        Molecule.__init__(self, molden_out[0], molden_out[1])
        self.freqs = molden_out[2]
        self.nfreqs = len(self.freqs)
        self.modes = molden_out[3]

    @staticmethod
    def read_molden(fname: str) -> tuple[list, npt.NDArray, npt.NDArray, npt.NDArray]:
        """
        A method reads vibrational information from molden files - overrides  method in Molecule parent class
        :param fname: name of molden file to load coordinate and frequency information from
        :type fname: str
        :return: atomic labels, cartesian coordinates, frequencies (cm^-1) and normal mode coordinates
        :rtype: tuple
        """
        ftype = fname.split('.')[-1]
        if ftype != 'molden':
            raise Exception("File type must be molden")

        labels, atoms, freqs, vibs = [], [], [], []

        mfile = open(fname, 'r')
        Atoms = False
        FREQ = False
        FRNORMCOORD = False

        actvib = -1
        for line in mfile:
            # what section are we in
            if '[' in line or '--' in line:
                Atoms = False
                FREQ = False
                FRNORMCOORD = False

            if '[Atoms]' in line:
                Atoms = True
            elif '[FREQ]' in line:
                FREQ = True
            elif '[FR-NORM-COORD]' in line:
                FRNORMCOORD = True
                # extract the information in that section
            elif Atoms:
                words = line.split()
                labels += words[0]
                atom_vec = words[3:6]
                atoms += [[eval(coords) for coords in atom_vec]]
            elif FREQ:
                freqs += [eval(line)]
            elif FRNORMCOORD:
                if 'vibration' in line or 'Vibration' in line:
                    vib_list = []
                    actvib += 1
                    if actvib > -1:
                        vibs += [vib_list]
                else:
                    vib_list += [[eval(coor) for coor in line.split()]]

        freqs = np.array(freqs)
        vibs = np.array(vibs)
        atoms = np.array(atoms)
        return (labels, atoms, freqs, vibs)

#    def __check_freqs(self, freqs, vibs):
#        nfreq = len(freqs)
#        if nfreq != (3* self.natoms) and np.count_nonzero(freqs) != (3*self.natoms-5 or 3*self.natoms-6):
#            raise Exception("Must have 3*natom frequencies and 3*natom-5/6 normal modes")
#        dims = np.shape(vibs)
#        if dims != (nfreq, self.natoms, 3):
#            raise Exception("Vibrational modes dimensions do not match")


    def ret_normal_mode_matrix(self, mass_weight=False) -> npt.NDArray:
        """
        A method to calculate and return the pseudo inverse of the square matrix (nfreqs, 3*natom) that
        defines the collective normal modes. Can be optionally mass weighted.
        :param mass_weight: flag to mass weight the normal modes
        :type mass_weight: bool
        :return: normal mode matrix - rows define the modes and columns the atom coordinates
        :rtype: numpy.ndarray
        """
        nm_mat = np.reshape(self.modes, (self.nfreqs, 3*self.natoms))
        nm_mat = np.linalg.pinv(nm_mat)
        if mass_weight:
            mass_mat = self.__mass_matrix(self.Zs, self.nfreqs, self.natoms)
            nm_mat = np.dot(mass_mat, nm_mat)
        return nm_mat

    @staticmethod
    def __mass_matrix(atomic_masses: list, nfreqs: int, natoms: int) -> npt.NDArray:
        mw_atom_vec = np.array([a ** 0.5 for a in atomic_masses for i in range(3)])
        mass_mat = np.eye(nfreqs, natoms * 3) * mw_atom_vec
        return mass_mat



class Atom:
    """
    A class to describe an Atom (not really used in current implementation of module)

    Attributes
    ----------
    self.label: str
        Atomic symbol
    self.vector: numpy.ndarray
        Vector of atom coordinates
    self.Z: float
        Atomic mass
    self.nelec : int
        Number of electrons
    """

    def __init__(self, label: str, vector: npt.NDArray):
        """
        Constructor for Atom object
        :param label: Atomic symbol
        :type label: str
        :param vector: vector of atomic coordinates
        :type vector: numpy.ndarray
        """
        self._label = label
        self.vector = vector
        self.__lookup_info()


    def __lookup_info(self):
        atom_info = periodic_table[self._label]
        self._Z = atom_info['Z']
        self._nelec = atom_info['nelec']

    @property
    def label(self):
        return self._label

    @label.setter
    def label(self, value):
        if type(value) != str:
            raise Exception("Atomic labels must be a string")
        else:
            value = value.upper()
            self._label = value
            self.__lookup_info()
            # lookup info and set z/nelec

    @property
    def Z(self):
        return self._Z

    @property
    def nelec(self):
        return self._nelec


class InternalCoordinate:
    """
    A class to describe internal coordinates

    Attributes
    ----------
    self.connectivity: list
        connectivity of the internal coordinate
    self.magnitude: float
        magnitude of the internal coordinate (bond length, angle, dihedral)
    """
    def __init__(self, connectivity: list, magnitude: float):
        self.connectivity = connectivity
        self.magnitude = magnitude

class BondDistance(InternalCoordinate):
    """
    A class to describe a bond length

    Attributes
    ----------
    self.connectivity: list
        the number of the two atoms that are bonded
    self.magnitude: float
        distance between the two atoms
    """
    def __init__(self, connectivity, distance):
        if len(connectivity) != 2:
            raise Exception("Bond lengths are defined between two atoms")
        InternalCoordinate.__init__(self, connectivity, distance)

class BondAngle(InternalCoordinate):
    """
    A class to describe a bond angle between two bonds

    Attributes
    ----------
    self.connectivity: list
        the numbers of the three atoms that make up the bond angle
    self.magnitude: float
        bond angle in degrees
    """
    def __init__(self, connectivity, angle):
        if len(connectivity) != 3:
            raise Exception("Bond angles are defined between three atoms/ two bond lengths")
        InternalCoordinate.__init__(self, connectivity, angle)

class DihedralAngle(InternalCoordinate):
    """
    A class to describe a dihedral angle between four atoms.
    The dihedral is defined as the angle between the planes of two bond lengths.

    Attributes
    ----------
    self.connectivity: list
        the numbers of the four atoms that make up the dihedral
    self.magnitude: float
        dihedral angle in degrees
    """
    def __init__(self, connectivity, angle):
        if len(connectivity) != 4:
            raise Exception("Dihedral angles are defined by four atoms - the angle between planes of two atom pairs")
        InternalCoordinate.__init__(self, connectivity, angle)


if __name__ == "__main__":
    # testing molecule class
    #molecule_path = 'data/Molecules/cs2.xyz'
    #mol = Molecule.init_from_xyz(molecule_path)
    #D = mol.distance_matrix()
    #mol.gen_internal_coords(D)
    #trajectory_path = 'data/Trajectories/CS2/output.xyz'
    #trj = SharcTrajectory.init_from_xyz(trajectory_path)
    #[bonds, angles, dihedrals] = trj.internal_coordinates()

    freq_path = 'data/Freq/freq.molden'
    ref_structure = Vibration(freq_path)
    #norm_mode_matrix = ref_structure.ret_normal_mode_matrix(mass_weight=True)

    trajectory_path = 'data/Trajectories/CS2/output.xyz'
    trj = SharcTrajectory.init_from_xyz(trajectory_path)
    trj.norm_mode_transform(ref_structure, mass_weighted=True)
    [avg, std] = trj.nma_analysis([[0, 2001]])

    print('Testing Done')