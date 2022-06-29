from setuptools import setup, find_packages

setup(
    name = 'pyqofta',

    version = '0.3.0',

    description = 'A python package for clustering quantum dynamcis simulations and reconstructing experimental observables',


    packages=find_packages("."),



#    packages = ['trajectories',
#                'molecules',
#  ],

    author = 'Kyle Acheson',
    author_email = '---',

    #long_description = open('README.md').read() + '\n\n' + open('CHANGELOG.md').read(),
    long_description = open('README.md').read(),
    long_description_content_type = "text/markdown",

    url='https://github.com/KyleAcheson/PyQOFTA',

    include_package_data=True,

    classifiers  = [
        'Development Status :: 1',
        'Programming Language :: Python :: 3',
        "License :: OSI Approved :: BSD License",
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Chemistry',
        'Topic :: Scientific/Engineering :: Physics',
    ],

    python_requires=">=3.7",


    install_requires = [

        'numpy ~= 1.20',
        'scipy ~= 1.8'

    ],


    keywords = [
        'Quantum Dynamics',
        'Clustering',
        'Trajectory Fitting',
        'Photochemistry',
        'Non-adiabatic Dynamics',
        'Inversion',
        'X-ray Scattering',
        'Electron Scattering',
        'Ultrafast Electron Diffraction'],

)
