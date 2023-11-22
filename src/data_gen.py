"""
Implementations for the data generators used in the experiments.
Currently implemented in tensorflow. Ideally these would be implemented in lab?
"""

import h5py

def print_hdf_structure(file_name):
    """
    Print the structure of an hdf5 file
    """
    with h5py.File(file_name, 'r') as f:
        f.visit(print)

