# PyQOFTA

A module for analysing trajectories, fitting to experimental observables and clustering.

### Trajectory Analysis:
* Initialise trajectories from xyz files
* Calculate internal coordinates
* Normal mode analysis (must provide reference modes to project on in molden format)

### Observables:
* X-ray scattering (elastic)
* Electron scattering (elastic)
* Total scattering (TBC)

### Observable Reconstruction/ Fitting:
(see https://doi.org/10.1039/D2CP01268E) 
* X-ray and electron scattering
* Optimise an ensemble of trajectory weights to fit the experimental observable
* Distill out contributions to observable
* Based on shotgun approach to direct minimisation of target function

### Clustering (TBC):
* Cluster trajectories with similar sampling of phase space
* Types of similarity measures...
* Types of clustering...
* Reconstruct observables with reduced dimensionality models


## Install

Ensure you are running >= python3.7

### Virtual Environment

First you must install `virtualenvwrapper`, which will allow you to build a
python environment that does not create conflicts with your main python installation.
To install use the following command:

```
pip3 install virtualenvwrapper
```

Then add the following to either your `~/.profile`, `~/.bash_profile`, or if running zsh `~/.zshrc`:

```
# virtualenvwrapper
# if you have python2 installed, you might need:
export VIRTUALENVWRAPPER_PYTHON=$(which python3)
# location of virtualenvwrapper.sh
source $(which virtualenvwrapper.sh)
# where to store the files for each venv
export WORKON_HOME=$HOME/.virtualenvs
# used when creating new projects
export PROJECT_HOME=$HOME/name_of_project_directory
```

Then run `source ~/.profile`/ `source ~/.bash_profile`/ `source ~/.zshrc` (whatever one you require for your setup)


You can now create new venvs using:

```
mkvirtualenv <name>
```

Trying running `pip3 list` and see that you only have the defualt modules installed.
You can install the modules you like and once you exit the venv you will default back to your main python environment. Exit the venv with:

```
decativate
```

and restart the venv with:

```
workon <name>
```

Now you have venvs installed, we can build the python package...

### Install PyQOFTA:

clone this repo using:

```
git clone https://github.com/KyleAcheson/PyQOFTA.git 
```

setup a new venv for development:

```
mkvirtualenv <name>
```

run the setup script which will install all dependancies and add PyQOFTA to your path:

```
python3 setup.py install --record installed_files.txt
```

here the `--record installed_files.txt` argument records where all the package files are installed.
If you wish to uninstall PyQOFTA, run:

```
sudo rm $(cat installed_files.txt)
```

### Generating documentation:

install pdoc3 if not installed:
```
pip3 install pdoc3
```

to generate a local server to view documentation of the modules run the following command in
the root directory of the repo
```
pdoc3 --http localhost:8080 pyqofta
```

go to:
```
http://localhost:8080
```

currently an issue with restructured text formatted docstrings - to fix
