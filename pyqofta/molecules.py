"""
Author: Kyle Acheson

A Module containing the basic molecular data structures used within the pyqofta framework.
These structures include `Molecule`, `Vibration` and `InternalCoordinates`.
Inside each are a series a routines for instantiating objects from file types such as `xyz` and `molden`.
These structures are used within other external modules such as `scattering.py` and `normal_mode_analysis.py`
to perform various operations on these structures. These structures are also used within the module `trajectories.py`
which contain data structures that define trajectories and ensembles of trajectories using these molecular
structure as building blocks.

If an atom required is not parameterised, you may add it to the dictionary in the source code: `periodic_table`.
"""

import numpy as np
import numpy.typing as npt
from scipy.sparse import csr_matrix, triu, dok_matrix

__all__ = [
    'Molecule',
    'Vibration',
    'InternalCoordinates',
]

class MoleculeTypeError(TypeError):
    def __init__(self, msg='Requires a Molecule object', *args, **kwargs):
        super().__init__(msg, *args, **kwargs)

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
    A static molecule object. Includes all information on a molecular structure at a given point
    in time along a trajectory. Currently does not include any values of momentum, but can be extended to include.
    The structure can be built in a script by the user using its constructor which requires a list of atomic labels
    for each atom and a set of molecular coordinates stored as a numpy.ndarray where the rows corrospond to the atom number.
    Alternatively (as is the norm), one can construct the object using the class constructors `init_from_xyz`
    and `init_from_molden` which reads in coordinates and atomic labels from an external xyz or molden file.

    One can calculate all, or a selection of the internal coordinates for the molecule using the internal coordinate methods.
    This class also includes methods for calculating the rmsd between two molecule structures via. the Kabsch algorithm.
    It is also possible to calculate observables for the static molecule, although currently only using the scattering
    module. With the addition of fitting modules, it will be possible to fit an observable from a single static
    structure to an experimental observable.

    Attributes
    ----------
    atom_labels : list
        a list of strings containing atomic symbols for each atom in the molecule - forced to be upper case.
    coordinates : numpy.ndarray
        molecule cartesian coordinates - rows corrosond to the atoms and columns xyz.
    natoms : int
        total number of atoms in molecule.
    nelecs : list
        list of the number of electrons for each atom entry of the coordinate matrix.
    Zs : list
        list of the atomic masses for each atom entry of the coordinate matrix .
    momenta : numpy.ndarray
        momentum vectors in cartesian coordinates for each atom (set to None by default).
    """

    def __init__(self, labels, coordinates, momenta=None):
        self._atom_labels = self.__check_atom_labels(labels)
        self.natoms, self.nelecs, self.Zs = self.__get_atoms_info()
        self._coordinates = coordinates
        self._momenta = momenta

    def __repr__(self):
        return f"Molecule({self._atom_labels}, {self.natoms}, {self.nelecs}, {self.Zs}, {self._coordinates.__repr__()}, {self._momenta})"

    def __iter__(self):
        return MoleculeIterator(self)


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
        and has the correct type and dimensionality.
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
        A class constructor that instantiates a Molecule object from an external xyz file specified in the path.

        Parameters
        ----------
        fpath : str
            absolute path to the xyz file containing the molecular structure
        """
        atom_names, coords = cls.read_xyz_mol(fpath)
        return cls(
            labels=atom_names,
            coordinates=coords
        )

    @staticmethod
    def read_xyz_mol(fname: str) -> tuple[list, npt.NDArray]:
        """
        A static method which reads coordinates from an xyz file.

        Parameters
        ----------
        fname : str
            path to file.

        Returns
        -------
        labels: list
            a list of atomic labels for each atom in the coordinate matrix.
        atom_coords: numpy.ndarray
            an array that corrosponds to cartesian coordinates, rows are the atoms.

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
        A class constructor that instantiates a Molecule object from an external molden file.

        Parameters
        ----------
        fpath : str
            path to the molden file
        """
        atom_labels, coords = cls.read_molden(fpath)
        return cls(
            labels=atom_labels,
            coordinates=coords
        )

    @staticmethod
    def read_molden(fname: str) -> tuple[list, npt.NDArray]:
        """
        A static method which reads coordinates a molden file.

        Parameters
        ----------
        fname : str
            path to file.

        Returns
        -------
        labels: list
            a list of atomic labels for each atom in the coordinate matrix.
        atom_coords: numpy.ndarray
            an array that corrosponds to cartesian coordinates, rows are the atoms.
        """
        ftype = fname.split('.')[-1]
        if ftype != 'molden':
            raise MoldenTypeError('Molecule.read_molden() requires a file in .molden format')

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

    def distance_matrix(self):
        """
        A method to calculate the distance matrix of a given structure.

        Returns
        -------
        dist_mat : scipy.sparse._csr_csr_matrix
            upper triangular distance matrix stored in sparse format (see scipy docs).
            To convert to a numpy.ndarray use the method `dist_mat.toarray()`.
        """
        dist_mat = dok_matrix((self.natoms, self.natoms), dtype=np.float64)  # more efficient for building mat
        for i in range(self.natoms):
            for j in range(i + 1, self.natoms):
                rvec = self.coordinates[i, :] - self.coordinates[j, :]
                dist_mat[i, j] = np.linalg.norm(rvec)
        dist_mat = triu(dist_mat, format='csr')  # convert to upper triangle matrix
        return dist_mat

    def bond_length(self, bond_connectivity: list) -> float:
        """
        A method to calculate the bond length between a pair of atoms in the self.

        Parameters
        ----------
        bond_connectivity : list
            a list that contains the indexes of the pairs of atoms in the coordinate matrix.
            `len(bond_connectivity) == 2`

        Returns
        -------
        bond_len : float
            value of the bond length in angstrom
        """
        rvec = self.coordinates[bond_connectivity[0], :] - self.coordinates[bond_connectivity[1], :]
        bond_len = np.linalg.norm(rvec)
        return bond_len

    def angle(self, angle_connectivity: list) -> float:
        """
        A method to calculate the angle between a pair of bond lengths R_ij and R_kj.

        Parameters
        ----------
        angle_connectivity : list
            list that contains the indexes of the atoms that make up the angle.
            `len(angle_connectivity) = 3`, where the second index is the central atom i.e. j

        Returns
        -------
        theta : float
            the bond angle in degrees
        """

        if len(angle_connectivity) != 3 or self.natoms < 3:
            raise AngleDefError
        i, j, k = angle_connectivity
        r_ij = self.coordinates[i, :] - self.coordinates[j, :]
        r_kj = self.coordinates[k, :] - self.coordinates[j, :]
        cosine_theta = np.dot(r_ij, r_kj)
        sin_theta = np.linalg.norm(np.cross(r_ij, r_kj))
        theta = np.arctan2(sin_theta, cosine_theta)
        theta = 180.0 * theta / np.pi
        return theta

    def dihedral(self, dihedral_connectivity: list) -> float:
        """
        A method to calculate the dihedral angle between two bond lengths that form a plane.

        Parameters
        ----------
        dihedral_connectivity : list
            connectivity of the atoms that form the dihedral angle (i, j, k, l).
            `len(dihedral_connectivity) = 4`

        Returns
        -------
        phi: float
            dihedral angle in degrees

        """
        if len(dihedral_connectivity) != 4 or self.natoms < 4:
            raise DihedralDefError
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

    def centre_of_mass(self):
        """
        A function to calculate the centre of mass in cartesian space. Take away the resulting vector
        from a molecule to shift the origin to the centre of mass.

        Returns
        -------
        centre_mass: numpy.ndarray
            centre of mass in cartesian coordinates
        """
        tot = np.zeros((1, 3))
        for i in range(self.natoms):
            tot = tot + self.Zs[i] * self.coordinates[i, :]
        centre_mass = tot / np.sum(self.Zs)
        return centre_mass

    def calculate_internal_coords(self):
        """
        A method to calculate *all* internal coordinates that describe the molecular structure.
        If the user is only interested in a selection of internal coordinates, for example one bond length or angle etc.,
        it is recommended to call the `bond_length`, `angle` or `dihedral` methods directly by specifying the atom
        connectivity.

        Returns
        -------
        internal_coords: InternalCoordinates
            a set of internal coordinates described by an InternalCoordinates object. This contains
            an attribute for each IC and the list of connectivities for each. In the case that of di- and tri-atomic
            molecules that have no defined angle or dihedral, those instance variables will be empty.

        """
        bond_lengths, angles, dihedrals = [], [], []
        bond_connectivities, angle_connectivities, dihedral_connectivities = [], [], []
        if self.natoms > 1: # only one bond length
            bond_connectivity = [0, 1] # first bond connecitivty = atom 0 and 1
            r = self.bond_length(bond_connectivity) # get bond length for given atoms
            bond_lengths.append(r) # append tuple of connecitvity and bond length
            bond_connectivities.append(bond_connectivity)
        if self.natoms > 2: # two bond lengths and one angle
            bond_connectivity = [0, 2]
            r = self.bond_length(bond_connectivity)
            bond_lengths.append(r)
            bond_connectivities.append(bond_connectivity)
            ang_connectivity = [2, 0, 1] # first angle is centred on atom 0 and between bonds r_01 and r_02
            ang = self.angle(ang_connectivity) # get angle for given atoms
            angles.append(ang) # append tuple
            angle_connectivities.append(ang_connectivity)
        if self.natoms > 3: # n-1 bond lengths, n-2 angles and n-3 dihedrals
            for i in range(3, self.natoms):
                bond_connectivity = [i - 3, 1]
                r = self.bond_length(bond_connectivity)
                bond_lengths.append(r)
                bond_connectivities.append(bond_connectivity)
                ang_connectivity = [i, i - 3, i - 2]
                ang = self.angle(ang_connectivity)
                angles.append(ang)
                angle_connectivities.append(ang_connectivity)
                dihedral_connectivity = [i, i - 3, i - 2, i - 1] # dihedral connectivity for molecules > 3 atoms
                dih = self.dihedral(dihedral_connectivity) # get dihedral
                dihedrals.append(dih) # append tuple
                dihedral_connectivities.append(dihedral_connectivity)
        return InternalCoordinates(bond_lengths,
                                   bond_connectivities,
                                   angles,
                                   angle_connectivities,
                                   dihedrals,
                                   dihedral_connectivities)

    @staticmethod
    def Kabsch_rmsd(molecule, referance_structure, Hydrogens=True):
        """
        A method to calculate the minimum RMSD between two geometries through the Kabsch algorithm.
        This works by calculating the centroid of each vector X (i.e. `sum(x)/ len(x)`) and aligning the two
        geometries. Then by calculating the covariance matrix of the two centred structures, this is used
        to calculate the rotation matrix that minimises the rmsd through a procedure based on single value decomposition.

        Parameters
        ----------
        referance_structure : Molecule
            the seconnd molecule to with which the RMSD is calculated wrt the instance of the molecule objecy

        Returns
        -------
        lrms: float
            the lowest possible RMSD between the structures (after rotation and translation)
        """

        if not isinstance(referance_structure, Molecule):
            raise MoleculeTypeError('Kabsch algorithm requires the reference to be another molecular structure')
        if molecule.natoms != referance_structure.natoms:
            raise MoleculeTypeError('The two molecules must have the same dimensions')
        if not Hydrogens:
            molecule, referance_structure = Molecule.__remove_hydrogens(molecule, referance_structure)
            print(molecule)
        nc = np.shape(molecule.coordinates)[1]
        p0 = np.sum(molecule.coordinates, axis=0)/molecule.natoms
        q0 = np.sum(referance_structure.coordinates, axis=0)/referance_structure.natoms
        geom1 = molecule.coordinates - p0
        geom2 = referance_structure.coordinates - q0 # translate coords to align centroid w origin
        cov = np.transpose(geom1) @ geom2 # calculate covariance matrix
        v, s, wh = np.linalg.svd(cov) # do single value decomp. on covariance matrix
        w = wh.T
        w = np.squeeze(w)
        v = np.squeeze(v)
        eye = np.eye(nc) # init identity matrix
        if np.linalg.det(w @ np.transpose(v)) < 0:
            eye[nc - 1, nc - 1] = -1
        u = w @ eye @ np.transpose(v) # rotation matrix that minimises the rmsd
        for i in range(molecule.natoms):
            geom1[i, :] = u @ geom1[i, :]
        diff = geom1 - geom2
        lrms = np.sqrt((np.sum(diff**2))/ molecule.natoms)
        return float(lrms)

    @staticmethod
    def __remove_hydrogens(mol1, mol2):
        mol_a, mol_b, labels_a, labels_b = [], [], [], []
        for i in range(mol1.natoms):
            if mol1.atom_labels[i] != 'H':
                mol_a.append(mol1.coordinates[i, :])
                labels_a.append(mol1.atom_labels[i])
            if mol2.atom_labels[i] != 'H':
                mol_b.append(mol2.coordinates[i, :])
                labels_b.append(mol2.atom_labels[i])
            else:
                pass
        new_mol_a = Molecule(labels_a, np.array(mol_a))
        new_mol_b = Molecule(labels_b, np.array(mol_b))
        return new_mol_a, new_mol_b





class Vibration(Molecule):
    """
    Class to represent vibrational modes of a molecule structure. Inherets properties and methods of Molecule.
    Stores information on normal modes, frequencies and the reference molecular structure.
    Is instantiated as `Vibration(fpath)` from a molden file path containing a frequencey calculation performed using
    the electronic structure code of choice.

    This class is typically used in normal mode analysis (see module `normal_mode_analysis` and example scripts
    `single_traj_nma.py`/ `ensemble_nma.py`). One can define a reference vibrational structure to project
    a trajectory or an ensemble of trajectories onto.

    Attributes
    ----------
    self.freqs : list
        vibrational frequencies for each of the 3N modes (in cm^-1)
    self.nfreqs : int
        Number of vibrational frequencies
    self.modes : nump.ndarray
        vectors of each of the normal mode frequencies
    """

    def __init__(self, fpath: str):
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
        A method to read coordinates, frequencies and normal modes from a molden files.
        Parameters
        ----------
        fname : str
            absolute path to molden file

        Returns
        -------
        labels : list
            a list of atom labels for each entry in the reference structures cartesian coordinate matrix
        atoms : numpy.ndarray
            array of coordinates for each atom in the reference structure
        freqs : numpy.ndarray
            the values of the frequencies of each mode in cm^-1
        vibs : numpy.ndarray
            the normal modes of each of the frequencies in contained in `freqs`
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

    @staticmethod
    def freq2time(freq_wavenum: float):
        """
        Convert a vibrational freq given in cm^-1 to a time period in fs

        Parameters
        ----------
        freq_wavenum : float
            vibrational freq in inverse cm

        Returns
        -------
        time_period: float
            period of the vibration in femtoseconds

        """
        c = 2.998e+8 # in m/s
        lambd = 0.01 /freq_wavenum # wavelength in m
        time_period = 1/(c/lambd) # period of vibration in time (s)
        return time_period*1e15 # convert to fs





#    def __check_freqs(self, freqs, vibs):
#        nfreq = len(freqs)
#        if nfreq != (3* self.natoms) and np.count_nonzero(freqs) != (3*self.natoms-5 or 3*self.natoms-6):
#            raise Exception("Must have 3*natom frequencies and 3*natom-5/6 normal modes")
#        dims = np.shape(vibs)
#        if dims != (nfreq, self.natoms, 3):
#            raise Exception("Vibrational modes dimensions do not match")



class InternalCoordinates:
    """
    Class to structure all internal coordinates of a molecule/ trajectory. Typically, only used within the framework of
    calculate *all* internal coordinates. One can access for example, only the bond lengths, with the property
    `instance_name.bonds`, allowing the user to distinguish between the different internal coordinates generated with ease.

    This class is called in both the `Trajectory` and `Molecule` internal coordinate routines. In the event that
    one calls `trajectory.calculate_internal_coords()` the bond lengths/ angles/ dihedrals will be type numpy.ndarray
    with dimensions [traj.nts, number of lengths/ angles]. In the event `molecule.calulcate_internal_coords()` is invoked,
    the numpy.ndarray is 1D. Atom connectivities are always returned as a list of lists.

    Attributes:
    -----------
    self.bonds: numpy.ndarray
        list of all bond lengths in angstroms
    self.bond_connectivity: list
        list of each present bond lengths atomic connectivity
    self.angles: numpy.ndarray
        list of all angles in degrees
    self.angle_connectivity: list
        list of all present angles atomic connectivity
    self.dihedrals: numpy.ndarray
        list of all dihedral angles in degrees
    self.dihedral_connectivity: list
        list of all present dihedral angles connectivity
    """

    def __init__(self, bonds, bond_connectivity, angles, angle_connectivity, dihedrals, dihedral_connectivity):
        self.bonds = np.array(bonds)
        self.bond_connectivity = bond_connectivity
        self.angles = np.array(angles)
        self.angle_connectivity = angle_connectivity
        self.dihedrals = np.array(dihedrals)
        self.dihedral_connectivity = dihedral_connectivity


class MoleculeIterator:

    def __init__(self, molecule):
        self._molecule = molecule
        self._index = 0

    def __next__(self):
        if self._index < len(self._molecule.coordinates):
            result = (self._molecule.atom_labels[self._index], self._molecule.coordinates[self._index, :])
            self._index += 1
            return result
        else:
            raise StopIteration




if __name__ == "__main__":
    # some testing stuff
    import pyqofta.trajectories as trj
    import matplotlib.pyplot as plt
    import os
    from natsort import natsorted
    traj_parent_dir = '/users/kyleacheson/CHD_TRAJS/'
    traj_paths = natsorted([traj_parent_dir + fpath for fpath in os.listdir(traj_parent_dir) if 'xyz' in fpath])  # list and sort all files numerically
    ensemble = trj.Ensemble.load_ensemble(traj_paths, 'sh')

    traj = ensemble.trajs[0]
    mol = traj.geometries[0]

    #rmsd_h = traj.Kabsch_rmsd(mol)
    rmsd_noh = traj.Kabsch_rmsd(mol, Hydrogens=False)


    print('done')

