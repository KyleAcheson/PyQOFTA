# Release notes
All notable changes to this project will be documented in this file.
This project adheres to [Semantic Versioning](http://semver.org/).

## 0.1.0

- Added method `Kabsch_rmsd` to `Molecule` class to calculate the RMSD wrt another molecule
- Added method `Kabsch_rmsd` to `Trajectory` class to calculate RMSD over a whole trajectory wrt some reference geometry

## 0.2.0
- Added `average_` method to `Ensemble` class which returns an average `Trajectory` object
- Converted `distance_matrix`, `dihedral`, `angle`, and `bond_length` to instance methods
- `Kabsch_rmsd` now includes a flag to remove Hydrogen atoms from the calculation (by default they are included), set `Hydrogens=False` to exclude.
- The `Ensemble` class can now be instantiated from a list of trajectory objects and the maximum number of time steps the trajectories are run for.
- To instantiate an instant of `Ensemble` from file, one now has to use the class method `load_ensemble` which takes a list of file paths and the trajectory type
- Added a static method `freq2time` to `Vibration` class - allows one to convert a vibrational freq in cm^-1 to a time period in fs.

