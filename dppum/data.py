"""
Implementations for the data generators used in the experiments.
Currently implemented in tensorflow. Ideally these would be implemented in lab?
"""

import h5py
import numpy as np


def print_hdf_structure(file_name):
    """
    Print the structure of an hdf5 file
    """
    with h5py.File(file_name, 'r') as f:
        f.visit(print)


# %%


def hdf_get_metadata(filepath):
    """
    Return the metadata of an HDF5 file, which must be in the 'metadata' group.

    Parameters
    ----------
    filepath : str
        The path to the HDF5 file.

    Returns
    -------
    dict
        A dictionary containing the metadata. Each key-value pair in the
        dictionary corresponds to a name-item pair in the metadata.

    """
    metadata_dict = {}
    with h5py.File(filepath, 'r') as hf:
        # Add all items from the 'metadata' group to the dictionary
        if 'metadata' in hf:
            for name, item in hf['metadata'].attrs.items():
                # Convert numpy data types to native Python types
                if isinstance(item, (np.int64,np.uint8,np.float64)):
                    item = item.item()
                # Convert byte strings to Python strings
                elif isinstance(item, np.bytes_):
                    item = item.decode('utf-8')
                
                metadata_dict[name] = item
    return metadata_dict