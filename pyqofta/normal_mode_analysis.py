import numpy as np
import numpy.typing as npt
from pyqofta.molecules import Molecule, Vibration
from pyqofta.trajectories import Ensemble, Trajectory


__all__ = [
    'nm_analysis',
    'nma_ensemble',
    'nma_traj',
    'normal_mode_transform'
]

class VibrationalTypeError(TypeError):
    def __init__(self, msg='Requires a Vibrational object', *args, **kwargs):
        super().__init__(msg, *args, **kwargs)

class MoleculeTypeError(TypeError):
    def __init__(self, msg='Requires a Molecule object', *args, **kwargs):
        super().__init__(msg, *args, **kwargs)

class TrajectoryTypeError(TypeError):
    def __init__(self, msg='Requires a Trajectory object', *args, **kwargs):
        super().__init__(msg, *args, **kwargs)

class EnsembleTypeError(TypeError):
    def __init__(self, msg='Requires a Ensemble object', *args, **kwargs):
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


def nma_traj(trajectory, ref_structure, time_intervals: list) -> tuple[npt.NDArray, npt.NDArray]:
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
    if not isinstance(trajectory, Trajectory):
        raise TrajectoryTypeError('Normal mode analysis over a trajectory requires an instance of Trajectory')
    if not isinstance(ref_structure, Vibration):
        raise VibrationalTypeError('An instance of Vibration containing reference normal modes is required to project the trajectory onto')

    normal_mode_trajectory = Trajectory.broadcast(normal_mode_transform, trajectory, ref_structure)
    normal_mode_trajectory = np.array(list(normal_mode_trajectory)) # convert from map obj to array of nm coords

    ntints = len(time_intervals)
    interval_std = np.zeros((ntints, ref_structure.nfreqs))
    interval_avg = np.zeros((ntints, ref_structure.nfreqs))
    for i in range(ntints):
        time_interval = time_intervals[i]
        tstart, tend = time_interval[0], time_interval[1]
        tdiff = tend - tstart
        nm_coords = normal_mode_trajectory[tstart:tend, :] # select normal mode coords over specified time interval
        summed_over_tint = np.sum(nm_coords, axis=0)  # coords summed over time interval (used for std calculation)
        sq_summed_over_tint = np.sum(nm_coords**2, axis=0) # sum of the coords squared over time interval
        if tdiff != 0: # calculate standard dev. of normal modes over each time interval
            avg_tint = summed_over_tint / tdiff
            interval_avg[i, :] = avg_tint
            avg_sq_tint = sq_summed_over_tint / tdiff
            interval_std[i, :] = (tdiff / (tdiff - 1) * (avg_sq_tint - avg_tint ** 2)) ** .5
    return interval_avg, interval_std


def nma_ensemble(ensemble, ref_structure, time_intervals: list):
    """
    A function to perform normal mode analysis on a whole ensemble of trajectories given a reference Vibrational
    structure to project on. This relies on calculating the normal mode coordinates of the 'average' trajectory
    and thus only includes coherent motion. The standard deviation on this average gives an idea of how active
    each normal mode is in the dynamics. Time intervals can be provided to perform analysis over each of these independent
    intervals in time - useful if there is some periodicity.
    :param ensemble: an ensemble of trajectories
    :type ensemble: Ensemble
    :param ref_structure: a reference structure containing normal mode and freq information
    :type ref_structure: Vibration
    :param time_intervals: intervals in time over which to perform analysis
    :type time_intervals: list (of lists)
    :return: average normal modes and standard deviation over all time, and the average + std. dev. in each time interval
    :rtype: numpy.ndarray
    """
    if not isinstance(ensemble, Ensemble):
        raise EnsembleTypeError('Normal mode analysis over a ensemble of trajectories requires an instance of Ensemble')
    if not isinstance(ref_structure, Vibration):
        raise VibrationalTypeError('An instance of Vibration containing reference normal modes is required to project the trajectory onto')

    nm_sum, nmsq_sum = np.zeros((ensemble.nts_max, ref_structure.nfreqs)), np.zeros((ensemble.nts_max, ref_structure.nfreqs))
    time_count = np.zeros(ensemble.nts_max, dtype=np.int64)
    for tbf in ensemble:
        nm_tbf = Trajectory.broadcast(normal_mode_transform, tbf, ref_structure)
        nm_tbf = np.array(list(nm_tbf))
        nm_sum += nm_tbf
        nmsq_sum += nm_tbf**2  # TODO: ENSURE WORKS FOR TRAJS WITH DIFFERENT NTS
        time_count[:tbf.nts] += 1

    ensemble_std, avg_nm = np.zeros((ensemble.nts_max, ref_structure.nfreqs)), np.zeros((ensemble.nts_max, ref_structure.nfreqs))
    for ts in range(ensemble.nts_max):
        for v in range(ref_structure.nfreqs):
            avg_ts = nm_sum[ts, v] / time_count[ts] # normal mode coords of the average traj
            avg_sq_ts = nmsq_sum[ts, v]/ time_count[ts]
            ensemble_std[ts, v] = (time_count[ts] / (time_count[ts] - 1) * (avg_sq_ts - avg_ts ** 2)) ** 0.5 # std of norm mode coords wrt avg traj
            avg_nm[ts, v] = avg_ts # normal mode coords of 'average trajectory'

    # TODO: ADD TIME INTERVAL ANALYSIS
    return avg_nm, ensemble_std


def nm_analysis(obj, ref_structure, time_intervals):
    """
    A wrapper for nma_traj and nma_ensemble functions.
    Will either return ensemble analysis on the average trajectory or individual trajectory based analysis,
    depending on the type of the input object.
    See documention for each of these functions for more information.
    """
    # TODO: ADD CENTRE OF MASS OPTION
    if isinstance(obj, Ensemble):
        [avg_normal_modes, ensemble_std] = nma_ensemble(obj, ref_structure, time_intervals)
        return avg_normal_modes, ensemble_std
    elif isinstance(obj, Trajectory):
        [interval_avg, interval_std] = nma_traj(obj, ref_structure, time_intervals)
        return interval_avg, interval_std
    else:
        raise TypeError('Normal mode analysis can only be conducted on a trajectory or ensemble')