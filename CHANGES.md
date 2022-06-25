# Release notes
All notable changes to this project will be documented in this file.
This project adheres to [Semantic Versioning](http://semver.org/).

## 0.1.0

- Added method `Kabsch_rmsd` to `Molecule` class to calculate the RMSD wrt another molecule
- Added method `Kabsch_rmsd` to `Trajectory` class to calculate RMSD over a whole trajectory wrt some reference geometry
- Added `average_` method to `Ensemble` class which returns the average trajectory of the ensemble as a list of `Molecule` types.
