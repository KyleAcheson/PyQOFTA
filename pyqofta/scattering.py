import numpy as np
import numpy.typing as npt
from pyqofta.molecules import Molecule
from pyqofta.trajectories import Ensemble, Trajectory, TrajectorySH

"""
Author: Kyle Acheson

A module for calculating scattering observables for individual molecules, trajectories and ensembles of trajectories.
So far limited to rotationally averaged elastic scattering within the IAM for Surface Hopping trajectories.
Will soon include elastic scattering for Gaussian wavepacket based methods within the BAT approximation for the
calculation of off-diagonal terms between different trajectories. An interface to Ab-initio scattering code may
one day be implemented.
"""

__all__ = [
    'IAM_ensemble_scattering',
    'IAM_trajectory_scattering',
    'IAM_molecular_scattering',
    'IAM_form_factors'
]

class FormFactorParameterisationError(ValueError):
    def __init__(self, msg='Form factor for the requested atom needs to be paramteterised', *args, **kwargs):
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



IAM_factors_dict = {'H': {'a': [0.493002, 0.322912, 0.140191, 0.040810], 'b': [10.5109, 26.1257, 3.14236, 57.7997], 'c': 0.003038},
                    'C': {'a': [2.26069, 1.56165, 1.05075, 0.839259], 'b': [22.6907, 0.656665, 9.75618, 55.5949], 'c': 0.286977},
                    'S': {'a': [6.90530, 5.20340, 1.43790, 1.58630], 'b': [1.46790, 22.2151, 0.253600, 56.1720], 'c': 0.866900}
                    }


def IAM_ensemble_scattering(ensemble, qvec: npt.NDArray, fq: npt.NDArray, FF: npt.NDArray, ELEC=False) -> list:
    """
    Calculates the IAM elastic scattering signal for a whole ensemble of trajectories.

    Parameters
    ----------
    ensemble : Ensemble
        an ensemble of trajectories
    qvec : numpy.ndarray
        momentum transfer vector
    fq : numpy.ndarray
        atomic form factor for each atom
    FF : numpy.ndarray
        form factor products
    ELEC : Bool
        flag to request electron scattering (calculates dsM as common in UED community) - default = False

    Returns
    -------
    Iens: numpy.ndarray
        a numpy.ndarry of dimensions [ensemble.nts_max, ensemble.ntrajs, Nq], this array is initialised
        as an array of zeros for the maximum time length of all trajectories. Hence, at each step the next trajectories
        signal is added to it. *If any trajectory has `trajectory.nts < ensemble.nts_max` then that index in the array
        Iens will be filled with zeros for values of `t > ensemble.nts_max`, therefore it is up to the user to
        perform any temporal averaging to account for this.* Note: one can use `ensemble.tcount` to do this.
        If the percentage difference signal is needed, that is left to the user, returned here is *Itot*.
    """
    #        A list of numpy.ndarrays with the scattering signal for each trajectory. It is returned as a list
    #        as the number of time steps in each trajectory may not be consistent, so it is up to the user to
    #        perform any averaging with this in mind. If the user wishs to calculate the percentage difference signal,
    #        that is left up to them to do externally.

    if not isinstance(ensemble, Ensemble):
        raise EnsembleTypeError('To calculate scattering over an ensemble an Ensemble object is required')
    #Iens = ensemble.broadcast(IAM_trajectory_scattering, qvec, fq, FF, ELEC)
    #return list(Iens)
    Nq = len(qvec)
    Iens = np.zeros((ensemble.nts_max, ensemble.ntrajs, Nq), dtype=float)
    for idx, traj in enumerate(ensemble):
        Itrj = IAM_trajectory_scattering(traj, qvec, fq, FF, ELEC)
        Iens[:traj.nts, idx, :] += Itrj
    return Iens



def IAM_trajectory_scattering(trajectory, qvec, fq, FF, ELEC=False) -> npt.NDArray:
    """
    Calculates IAM elastic scattering for a single trajectory.

    Parameters
    ----------

    trajectory : Trajectory
        a trajectory
    qvec : numpy.ndarray
        momentum transfer vector
    fq : numpy.ndarray
        atomic form factor for each atom
    FF : numpy.ndarray
        form factor products
    ELEC : Bool
        flag to request electron scattering (calculates dsM as common in UED community) - default = False

    Returns
    -------

    Itrj: numpy.ndarray
        scattering signal for a single trajectory
    """
    if not isinstance(trajectory, Trajectory):
        raise TrajectoryTypeError('To calculate the scattering over time a Trajectory object is required.')
    Itrj = trajectory.broadcast(IAM_molecular_scattering, qvec, fq, FF, ELEC)
    return np.array(list(Itrj))


def IAM_molecular_scattering(molecule, qvec, fq, FF, ELEC=False) -> npt.NDArray:
    """
    IAM scattering for a single static molecule

    Parameters
    ----------
    molecule : Molecule
        a static molecule object
    qvec : numpy.ndarray
        momentum transfer vector
    fq : numpy.ndarray
        atomic form factor for each atom
    FF : numpy.ndarray
        form factor products
    ELEC : Bool
        flag to request electron scattering (calculates dsM as common in UED community) - default = False

    Returns
    -------
    Itot : numpy.ndarray
        scattering signal from a single molecule (Itot for xrs and sM (q*Imol/Iat) for ued)
    """

    Nq = len(qvec)
    Imol = np.zeros(Nq)
    Iat = sum(fq**2)
    for i in range(molecule.natoms):
        for j in range(i+1, molecule.natoms):
            qr_ij = qvec * np.linalg.norm(molecule.coordinates[i, :] - molecule.coordinates[j, :])
            sin_qr_ij = np.sinc(qr_ij)
            Imol += 2 * FF[i, j, :] * sin_qr_ij
    if ELEC:
        Itot = (qvec * Imol)/Iat # sM (modified scattering used in ued community)
        return Itot
    else:
        Itot = Imol + Iat
        return Itot # standard xrs



def IAM_form_factors(molecule, qvec: npt.NDArray, ELEC=False) -> tuple[npt.NDArray, npt.NDArray]:
    """
    Gets the IAM form factors for a given molecule (either x-ray or electron), also calculates the form factor
    product that appear in the calculation - so you do not have to calculate them at each iteration of the loop.

    If `ELEC=True` then will return the Mott-Bethe form factors.
    TODO: ADD COMPATIBILITY WITH ELSEPA FORM FACTORS

    Parameters
    ----------

    molecule : Molecule
        a static molecule object
    qvec : numpy.ndarray
        momentum transfer vector
    ELEC : Bool
        flag to request electron scattering (calculates dsM as common in UED community) - default = False

    Returns
    -------
    fq : numpy.ndarray
        atomic form factor for each atom
    FF : numpy.ndarray
        form factor products
    """
    Nq = len(qvec)
    if not isinstance(molecule, Molecule):
        raise MoleculeTypeError('IAM form factor computation requires a Molecule object')
    if ELEC:
        fq = els(molecule.atom_labels, qvec)
    else:
        fq = xrs(molecule.atom_labels, qvec)

    FF = np.zeros((molecule.natoms, molecule.natoms, Nq))
    for i in range(molecule.natoms):
        for j in range(molecule.natoms):
            FF[i, j, :] = fq[i, :] * fq[j, :]
    return FF, fq



def xrs(atoms, qAng):
    """
    Returns the IAM form factors for xrs
    """

    Nat = len(atoms)
    Nq = len(qAng)
    fq = np.zeros((Nat, Nq))
    for i, atom in enumerate(atoms):
        try:
            tmp = np.zeros(Nq)
            for j in range(4):
                tmp = tmp + IAM_factors_dict[atom]['a'][j] * np.exp(-IAM_factors_dict[atom]['b'][j] * (qAng / (4 * np.pi)) ** 2)
            fq[i, :] = IAM_factors_dict[atom]['c'] + tmp #TODO: MUST BE A WAY TO IMPROVE THIS
        # [(sum([IAM_factors_dict[atom]['a'][j] * np.exp(-IAM_factors_dict[atom]['b'][j] * (qAng / 4 * np.pi) ** 2) for j in range(4)]) + IAM_form_factors[atom]['c']) for atom in atoms]
        except KeyError:
            raise FormFactorParameterisationError(f'Atom {atom} not parameterised - edit IAM_factors_dict dict using ITC data')
    return fq


def els(atoms, qAng):
    """
    Caclulates the Mott-Bethe electron scattering form factors within IAM
    """

    Nat = len(atoms)
    Nq = len(qAng)
    fq = np.zeros((Nat, Nq))
    for i, atom in enumerate(atoms):
        try:
            tmp = np.zeros(Nq)
            for j in range(4):
                tmp = tmp + IAM_factors_dict[atom]['a'][j] * np.exp(-IAM_factors_dict[atom]['b'][j] * (qAng / (4 * np.pi)) ** 2)
            fq[i, :] = (atom - (IAM_factors_dict[atom]['c'] + tmp)) / qAng ** 2
            # TODO: BUG HERE AS atom IS A LABEL - NO LONGER ATOMIC MASS - FIX
        except KeyError:
            raise FormFactorParameterisationError(f'Atom {atom} not parameterised - edit IAM_factors_dict dict using ITC data')
    return fq
