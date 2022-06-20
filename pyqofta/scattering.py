import numpy as np
import numpy.typing as npt
from pyqofta.molecules import Molecule
from pyqofta.trajectories import Ensemble, Trajectory, TrajectorySH

'''
Author: Kyle Acheson

A module for calculating scattering observables for individual molecules, trajectories and ensembles of trajectories.
So far limited to rotationally averaged elastic scattering within the IAM for Surface Hopping trajectories.
Will soon include elastic scattering for Gaussian wavepacket based methods within the BAT approximation for the
calculation of off-diagonal terms between different trajectories. An interface to Ab-initio scattering code may
one day be implemented.
'''

__all__ = [
    'IAM_scattering',
    'IAM_form_factors'
]

class FormFactorParameterisationError(ValueError):
    def __init__(self, msg='Form factor for the requested atom needs to be paramteterised', *args, **kwargs):
        super().__init__(msg, *args, **kwargs)

class MoleculeTypeError(TypeError):
    def __init__(self, msg='Requires a Molecule object', *args, **kwargs):
        super().__init__(msg, *args, **kwargs)


IAM_factors_dict = {'H': {'a': [0.493002, 0.322912, 0.140191, 0.040810], 'b': [10.5109, 26.1257, 3.14236, 57.7997], 'c': 0.003038},
                    'C': {'a': [2.26069, 1.56165, 1.05075, 0.839259], 'b': [22.6907, 0.656665, 9.75618, 55.5949], 'c': 0.286977},
                    'S': {'a': [6.90530, 5.20340, 1.43790, 1.58630], 'b': [1.46790, 22.2151, 0.253600, 56.1720], 'c': 0.866900}
                    }

def IAM_scattering(molecule, qvec, ELEC=False):
    pass #TODO: WRITE IAM FUNCTIONS FOR MOLECULE AND TRAJECTORY



def IAM_form_factors(molecule, qvec: npt.NDArray, ELEC=False) -> tuple[npt.NDArray, npt.NDArray]:
    """
    Gets the IAM form factors for a given molecule (either x-ray or electron), also calculates the form factor
    product that appear as diagonal terms - saves calculation of the diagonal terms at every time step.
    :param molecule: a molecular structure
    :type molecule: Molecule
    :param qvec: momentum transfer vector in inverse angstrom
    :type qvec: numpy.ndarray
    :param ELEC: flag to request electron scattering factors instead of x-ray (default=False)
    :type ELEC: Bool
    :return: form factor products (sum**2 amounts diagonal term in IAM equation) and form factors
    :rtype: numpy.ndarray
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
    Returns the x-ray scattering form factor for a given series of atoms.
    :param atoms: atmoic labels
    :type atoms: list
    :param qAng: momentum transfer vector
    :type qAng: numpy.ndarray
    :return: x-ray form factor
    :rtype: numpy.ndarray
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
    Returns the electron scattering (Mott-Bethe) form factor for a given series of atoms.
    :param atoms: atmoic labels
    :type atoms: list
    :param qAng: momentum transfer vector
    :type qAng: numpy.ndarray
    :return: Mott-Bethe form factor
    :rtype: numpy.ndarray
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
