import numpy as np
import numpy.typing as npt
from pyqofta.molecules import Molecule, Vibration
from pyqofta.trajectories import Ensemble, Trajectory, TrajectorySH

'''
Author: Kyle Acheson

A module form performing normal mode analysis on trajectories and ensembles of trajectories
'''

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



def normal_mode_matrix(vibrational_structure, mass_weight=True) -> npt.NDArray:
    """
    A function to calculate the pseudo inverse of the square matrix [nfreqs, 3*natom] that
    defines the collective normal modes. Can be optionally mass weighted. Requires an instance of `Vibration`.

    Parameters
    ----------

    vibrational_structure : Vibration
        a reference structure of vibration type which contains normal mode and frequencey information to project onto
    mass_weight : Bool
        a flag to allow for mass weighting of normal modes (default = True)

    Returns
    -------

    nm_mat : numpy.ndarray
        normal mode matrix - rows define normal modes and columns the atom coordinates - dimensions [nfreqs, 3*natom]

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


def normal_mode_transform(molecule, ref_structure, mass_weighted=True) -> npt.NDArray:
    """
    A function to transform a given molecule onto a set of normal modes of a reference structure.
    By default the coordinates are not mass weighted. Requires a `Molecule` and `Vibration` instance.

    Parameters
    ----------

    molecule : Molecule
        the molecule structure to project into normal mode coordinates from cartesian
    ref_structure : Vibration
        the reference structure containing normal mode information to project `molecule` onto
    mass_weighted : Bool
        allows for mass weighting (default = True)

    Returns
    -------

    norm_mode_coords: numpy.ndarray
        the input `molecule` transformed into normal mode basis wrt `ref_structure`
    """
    if not isinstance(molecule, Molecule):
        raise MoleculeTypeError('An instance of Molecule is required for the normal mode transform')
    if not isinstance(ref_structure, Vibration):
        raise VibrationalTypeError('An instance of Vibration is required to project the molecule onto')
    norm_mode_mat = normal_mode_matrix(ref_structure, mass_weighted)
    #if centre_mass:
    #    mcmass = molecule.centre_of_mass()
    #    rcmass = ref_structure.centre_of_mass()
    #    mcoords = molecule.coordinates - mcmass
    #    rcoords = ref_structure.coordinates - rcmass
    #else:
    mcoords = molecule.coordinates
    rcoords = ref_structure.coordinates
    displacement_vec = mcoords.flatten() - rcoords.flatten()
    normal_mode_coords = np.dot(displacement_vec, norm_mode_mat)
    return normal_mode_coords


def nma_traj(trajectory, ref_structure, time_intervals: list) -> tuple[npt.NDArray, npt.NDArray]:
    """
    A method for trajectory normal mode analysis. Calculates the average of the normal mode coordinates over
    a number of time intervals specified. Also calculates the standard deviation of these modes within the intervals.
    This gives an indicator of how active a given normal mode is within the trajectory.

    Parameters
    ----------

    trajectory : Trajectory
        the trajectory to perform nma on
    ref_structure : Vibration
        reference structure with freq/ normal mode information to project each timestep in `trajectory` onto
    time_intervals : list
        a series of time intervals - can just be the whole of time, or can specify a more informed
        selection of intervals that reflect the period of a vibration.


    Returns
    -------

    interval_avg: numpy.ndarray
        average of normal mode coordinates within the set of `time_intervals` provided
    interval_std: numpy.ndarray
        standard deviation on the average over the set of `time intervals`

    """
    if not isinstance(trajectory, Trajectory):
        raise TrajectoryTypeError('Normal mode analysis over a trajectory requires an instance of Trajectory')
    if not isinstance(ref_structure, Vibration):
        raise VibrationalTypeError('An instance of Vibration containing reference normal modes is required to project the trajectory onto')

    normal_mode_trajectory = trajectory.broadcast(normal_mode_transform, ref_structure)
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

    Parameters
    ----------

    ensemble : Ensemble
        an ensemble of trajectories to perform nma on
    ref_structure : Vibration
       reference structure with freq/ normal mode information to project each timestep within each trajectory onto
    time_intervals : list
        a series of time intervals - can just be the whole of time, or can specify a more informed
        selection of intervals that reflect the period of a vibration.

    Returns
    -------
    avg_nm : numpy.ndarray
        the average trajectories normal mode coordinates
    ensemble_std: numpy.ndarray
        standard deviation on the average trajectory in normal mode basis

    """
    if not isinstance(ensemble, Ensemble):
        raise EnsembleTypeError('Normal mode analysis over a ensemble of trajectories requires an instance of Ensemble')
    if not isinstance(ref_structure, Vibration):
        raise VibrationalTypeError('An instance of Vibration containing reference normal modes is required to project the trajectory onto')

    nm_sum, nmsq_sum = np.zeros((ensemble.nts_max, ref_structure.nfreqs)), np.zeros((ensemble.nts_max, ref_structure.nfreqs))
    time_count = np.zeros(ensemble.nts_max, dtype=np.int64)
    for tbf in ensemble:
        nm_tbf = tbf.broadcast(normal_mode_transform, ref_structure)
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


def nm_analysis(obj, ref_structure, time_intervals: list):
    """
    A wrapper for `nma_ensemble` and `nma_traj` - will call the relevent function depending on the input type of `obj`.
    See the documentation of these functions for more information.

    Parameters
    ----------
    obj : Trajectory or Ensemble
        either the trajectory or ensemble structure
    ref_structure : Vibration
        the reference structure with normal mode information
    time_intervals : list
        a series of time intervals - can just be the whole of time, or can specify a more informed
        selection of intervals that reflect the period of a vibration.

    Returns
    -------
    The return values of `nma_ensemble` or `nma_traj` - case dependent.

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


if __name__ == "__main__":
    traj_path = '/Users/kyleacheson/PycharmProjects/PyQOFTA/data/Trajectories/CS2/diss/TRAJ_00010/output.xyz'
    traj = TrajectorySH.init_from_xyz(traj_path)
    ref_structure = Vibration('/Users/kyleacheson/PycharmProjects/PyQOFTA/data/Freq/cs2.molden')
    [int_avg, int_std] = nma_traj(traj, ref_structure, [[0, 1000]])
    print('done')