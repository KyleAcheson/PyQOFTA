import numpy as np
import numpy.typing as npt
from molecules import Molecule, Vibration


class VibrationalTypeError(TypeError):
    def __init__(self, msg='Requires a Vibrational object', *args, **kwargs):
        super().__init__(msg, *args, **kwargs)

class MoleculeTypeError(TypeError):
    def __init__(self, msg='Requires a Molecule object', *args, **kwargs):
        super().__init__(msg, *args, **kwargs)

class TrajectoryTypeError(TypeError):
    def __init__(self, msg='Requires a Trajectory object', *args, **kwargs):
        super().__init__(msg, *args, **kwargs)


def normal_mode_matrix(vibrational_structure, mass_weight=False) -> npt.NDArray:
    """
    A function to calculate the pseudo inverse of the square matrix (nfreqs, 3*natom) that
    defines the collective normal modes. Can be optionally mass weighted.
    :param vibrational_structure: an instance of Vibration containing the required frequencies and vibrational mode coordinates
    :param mass_weight: flag to mass weight the normal modes
    :type mass_weight: bool
    :return: normal mode matrix - rows define the modes and columns the atom coordinates
    :rtype: numpy.ndarray
    """
    if not isinstance(vibrational_structure, Vibration):
        raise VibrationalTypeError('An instance of Vibration is required to calculate the normal mode matrix')
    nm_mat = np.reshape(vibrational_structure.modes, (vibrational_structure.nfreqs, 3 * vibrational_structure.natoms))
    nm_mat = np.linalg.pinv(nm_mat)
    if mass_weight:
        mw_atoms = np.array([atom_mass ** 0.5 for atom_mass in vibrational_structure.Zs for i in range(3)])
        mass_mat = np.eye(vibrational_structure.nfreqs, vibrational_structure.natoms * 3) * mw_atoms
        nm_mat = np.dot(mass_mat, nm_mat)
    return nm_mat


def normal_mode_transform(molecule, ref_structure, mass_weighted=False) -> npt.NDArray:
    """
    A function to transform a given molecule onto a set of normal modes of a reference structure.
    By default the coordinates are not mass weighted.
    :param molecule: an instance of Molecule containing the coordinates which are to be tranformed
    :type molecule: Molecule
    :param ref_structure: an instance of Vibration containing the set of normal modes onto which the molecule is projected
    :type ref_structure: Vibration
    :param mass_weighted: a flag to toggle if the coordinates are mass weight (default = False)
    :type mass_weighted: Bool
    :return: molecule projected into normal modes of reference Vibration (should be nfreqs of these)
    :rtype: numpy.ndarray
    """
    if not isinstance(molecule, Molecule):
        raise MoleculeTypeError('An instance of Molecule is required for the normal mode transform')
    if not isinstance(ref_structure, Vibration):
        raise VibrationalTypeError('An instance of Vibration is required to project the molecule onto')
    norm_mode_mat = normal_mode_matrix(ref_structure, mass_weighted)
    displacement_vec = molecule.coordinates.flatten() - ref_structure.coordinates.flatten()
    normal_mode_coords = np.dot(displacement_vec, norm_mode_mat)
    return normal_mode_coords


def nma_analysis(trajectory, time_intervals: list) -> tuple[npt.NDArray, npt.NDArray]:
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
    nvib = np.shape(trajectory.norm_mode_coords)[0]
    ntints = len(time_intervals)
    interval_std = np.zeros((nvib, ntints))
    interval_avg = np.zeros((nvib, ntints))
    for i in range(ntints):
        time_interval = time_intervals[i]
        tstart, tend = time_interval[0], time_interval[1]
        tdiff = tend - tstart
        nm_coords = trajectory.norm_mode_coords[:, tstart:tend] # select normal mode coords over specified time interval
        summed_over_tint = np.sum(nm_coords, axis=1)  # coords summed over time interval (used for std calculation)
        sq_summed_over_tint = np.sum(nm_coords**2, axis=1) # sum of the coords squared over time interval
        if tdiff != 0: # calculate standard dev. of normal modes over each time interval
            avg_tint = summed_over_tint / tdiff
            interval_avg[:, i] = avg_tint
            avg_sq_tint = sq_summed_over_tint / tdiff
            interval_std[:, i] = (tdiff / (tdiff - 1) * (avg_sq_tint - avg_tint ** 2)) ** .5
    return interval_avg, interval_std