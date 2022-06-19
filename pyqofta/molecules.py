'''
Author: Kyle Acheson

A module containing Molecule and Vibration

Objects of type Molecule can be instantiated without an external file, but
for the instantiation of an object of type Vibration one must provide frequency information
from an external molden file.

If an atom required is not parameterised, you may add it to the dictionary: periodic_table.

'''

import numpy as np
import numpy.typing as npt
from scipy.sparse import csr_matrix, triu, dok_matrix

__all__ = [
    'Molecule',
    'Vibration',
    'InternalCoordinates',
]

class AngleDefError(ValueError):
    def __init__(self, msg='Angles are defined between three atoms', *args, **kwargs):
        super().__init__(msg, *args, **kwargs)

class DihedralDefError(ValueError):
    def __init__(self, msg='Dihedrals are defined between two bond planes (4 atoms)', *args, **kwargs):
        super().__init__(msg, *args, **kwargs)

class XYZTypeError(TypeError):
    def __init__(self, msg='File type must be in .xyz format', *args, **kwargs):
        super().__init__(msg, *args, **kwargs)

class MoldenTypeError(TypeError):
    def __init__(self, msg='File type must be in .molden format', *args, **kwargs):
        super().__init__(msg, *args, **kwargs)


periodic_table = {
    'H': {'Z': 1.008, 'nelec': 1},
    'C': {'Z': 12.011, 'nelec': 6},
    'N': {'Z': 14.007, 'nelec': 7},
    'O': {'Z': 15.999, 'nelec': 8},
    'S': {'Z': 32.06, 'nelec': 16},
    'Br': {'Z': 79.904, 'nelec': 35},
    'I': {'Z': 126.90, 'nelec': 53}
}


class Molecule:
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

    Class methods:
    --------------
    init_from_xyz(cls, fpath: str)
    init_from_molden(cls, fpath: str)

    Instance methods:
    -----------------
    calculate_internal_coords()

    Static methods:
    ---------------
    read_xyz_mol(fname: str)
    read_molden(fname: str)
    distance_matrix(Molecule)
    bond_length(Molecule, bond_connectivity: list)
    angle(Molecule, angle_connectivity: list)
    dihedral(Molecule, dihedral_connectivity: list)
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

    def __repr__(self):
        return f'Molecule({self._atom_labels}, {self.natoms}, {self.nelecs}, {self.Zs}, {self._coordinates.__repr__()}, {self._momenta})'


    @property
    def coordinates(self):
        return self._coordinates

    @coordinates.setter
    def coordinates(self, coords: npt.NDArray):
        """
        Coordinate property is allowed to be set incase the instantiated coordinates are transformed.
        This setter ensures transformed coordinates are of same dimensionality.
        """
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
            raise XYZTypeError('Molecule.init_from_xyz() requires a file in .xyz format')
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
            raise MoldenTypeError('Molecule.init_from_molden() requires a file in .molden format')

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

    @staticmethod
    def distance_matrix(molecule):
        """
        Method that computes the distance matrix for a set of coordinates.
        Only returns upper triangle of matrix as a csr sparse matrix.
        :param molecule: a molecular structure
        :type molecule: Molecule
        :return: distance matrix
        :rtype: scipy.sparse._csr_csr_matrix
        """
        dist_mat = dok_matrix((molecule.natoms, molecule.natoms), dtype=np.float64)  # more efficient for building mat
        for i in range(molecule.natoms):
            for j in range(i + 1, molecule.natoms):
                rvec = molecule.coordinates[i, :] - molecule.coordinates[j, :]
                dist_mat[i, j] = np.linalg.norm(rvec)
        dist_mat = triu(dist_mat, format='csr')  # convert to upper triangle matrix
        return dist_mat

    @staticmethod
    def bond_length(molecule, bond_connectivity: list) -> float:
        """
        Calculate bond length between two atoms
        :param molecule: a molecule structure
        :type molecule: Molecule
        :param bond_connectivity: connectivity of two bonds in coordinate array
        :type bond_connectivity: list
        :return: bond length
        :rtype: numpy.float64
        """
        rvec = molecule.coordinates[bond_connectivity[0], :] - molecule.coordinates[bond_connectivity[1], :]
        return np.linalg.norm(rvec)

    @staticmethod
    def angle(molecule, angle_connectivity: list) -> float:
        """
        Caclulates angle between two bond lengths r_ij and r_kj where atom j is central
        :param molecule: a molecule structure
        :type molecule: Molecule
        :param angle_connectivity: Connectivity of atoms (i, j, k) that form an angle
        :type angle_connectivity: list
        :return: angle between atoms r_ij and r_kj in degrees
        :rtype: float
        """
        if len(angle_connectivity) != 3 or molecule.natoms < 3:
            raise AngleDefError
        i, j, k = angle_connectivity
        r_ij = molecule.coordinates[i, :] - molecule.coordinates[j, :]
        r_kj = molecule.coordinates[k, :] - molecule.coordinates[j, :]
        cosine_theta = np.dot(r_ij, r_kj)
        sin_theta = np.linalg.norm(np.cross(r_ij, r_kj))
        theta = np.arctan2(sin_theta, cosine_theta)
        theta = 180.0 * theta / np.pi
        return theta

    @staticmethod
    def dihedral(molecule, dihedral_connectivity: list) -> float:
        """
        Method to calculate a dihedral angle between two bond lengths that form a plane
        :param molecule: a molecular structure
        :type molecule: Molecule
        :param dihedral_connectivity: Connectivity of atoms that form a dihedral angle (i, j, k, l)
        :type dihedral_connectivity: list
        :return: dihedral angle (phi) in degrees
        :rtype: float
        """
        if len(dihedral_connectivity) != 4 or molecule.natoms < 4:
            raise DihedralDefError
        i, j, k, l = dihedral_connectivity
        r_ji = molecule.coordinates[j, :] - molecule.coordinates[i, :]
        r_kj = molecule.coordinates[k, :] - molecule.coordinates[j, :]
        r_lk = molecule.coordinates[l, :] - molecule.coordinates[k, :]
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

    # TODO: ADD COM METHOD TO CENTRE MOLECULE AT COM

    def calculate_internal_coords(self):
        """
        A method that calculates all possible internal coordinates
        (bond distances, angles, and dihedrals) as well as their connectivity.
        :return: Internal coordinates object containing bonds, angles and dihedrals + their connectivities
        :rtype: InternalCoordinates
        """
        bond_lengths, angles, dihedrals = [], [], []
        bond_connectivities, angle_connectivities, dihedral_connectivities = [], [], []
        if self.natoms > 1: # only one bond length
            bond_connectivity = [0, 1] # first bond connecitivty = atom 0 and 1
            r = self.bond_length(self, bond_connectivity) # get bond length for given atoms
            bond_lengths.append(r) # append tuple of connecitvity and bond length
            bond_connectivities.append(bond_connectivity)
        if self.natoms > 2: # two bond lengths and one angle
            bond_connectivity = [0, 2]
            r = self.bond_length(self, bond_connectivity)
            bond_lengths.append(r)
            bond_connectivities.append(bond_connectivity)
            ang_connectivity = [2, 0, 1] # first angle is centred on atom 0 and between bonds r_01 and r_02
            ang = self.angle(self, ang_connectivity) # get angle for given atoms
            angles.append(ang) # append tuple
            angle_connectivities.append(ang_connectivity)
        if self.natoms > 3: # n-1 bond lengths, n-2 angles and n-3 dihedrals
            for i in range(3, self.natoms):
                bond_connectivity = [i - 3, 1]
                r = self.bond_length(self, bond_connectivity)
                bond_lengths.append(r)
                bond_connectivities.append(bond_connectivity)
                ang_connectivity = [i, i - 3, i - 2]
                ang = self.angle(self, ang_connectivity)
                angles.append(ang)
                angle_connectivities.append(ang_connectivity)
                dihedral_connectivity = [i, i - 3, i - 2, i - 1] # dihedral connectivity for molecules > 3 atoms
                dih = self.dihedral(self, dihedral_connectivity) # get dihedral
                dihedrals.append(dih) # append tuple
                dihedral_connectivities.append(dihedral_connectivity)
        return InternalCoordinates(bond_lengths,
                                   bond_connectivities,
                                   angles,
                                   angle_connectivities,
                                   dihedrals,
                                   dihedral_connectivities)



class Vibration(Molecule):
    """
    Class to represent vibrational modes of a molecule structure - subclass of Molecule

    Attributes
    ----------
    self.freqs : list
        vibrational frequencies for each of the 3N modes (in cm^-1)
    self.nfreqs : int
        Number of vibrational frequencies
    self.modes : nump.ndarray
        vectors of each of the normal mode frequencies

    + All attributes inherited from Molecule class

    Methods
    _______

    Static Methods
    --------------
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

    def __repr__(self):
        return f'Vibration({self._atom_labels}, {self._coordinates}, {self.freqs})'

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
            raise MoldenTypeError('Vibration object must be instantiated from a .molden type file')

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



class InternalCoordinates:
    """
    Class to structure all internal coordinates of a molecule

    Attributes:
    -----------
    self.bonds: list
        list of all bond lengths in angstroms
    self.bond_connectivity: list
        list of each present bond lengths atomic connectivity
    self.angles: list
        list of all angles in degrees
    self.angle_connectivity: list
        list of all present angles atomic connectivity
    self.dihedrals: list
        list of all dihedral angles in degrees
    self.dihedral_connectivity: list
        list of all present dihedral angles connectivity
    """

    def __init__(self, bonds, bond_connectivity, angles, angle_connectivity, dihedrals, dihedral_connectivity):
        self.bonds = bonds
        self.bond_connectivity = bond_connectivity
        self.angles = angles
        self.angle_connectivity = angle_connectivity
        self.dihedrals = dihedrals
        self.dihedral_connectivity = dihedral_connectivity



if __name__ == "__main__":
    # some testing stuff
    print('done')

