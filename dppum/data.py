"""
Implementations for the data generators used in the experiments.
Currently implemented in tensorflow. Ideally these would be implemented in lab?
"""

import h5py
import tensorflow as tf
import pdb
import numpy


def print_hdf_structure(file_name):
    """
    Print the structure of an hdf5 file
    """
    with h5py.File(file_name, 'r') as f:
        f.visit(print)


# %%

def hdf_to_tf_dataset(filepath,dtype=tf.float32):
    """
    Load data from an HDF5 file and convert it into a TensorFlow dataset.
    The output dataset will have a batch size of 1. To create batches you
    will likely have to use data.padded_batch(minibatch_size).    

    Parameters
    ----------
    filepath : str
        The path to the HDF5 file.

    Returns
    -------
    data : tf.data.Dataset
        The TensorFlow dataset.

    """
    # Define a generator for the data
    def gen():
        with h5py.File(filepath, 'r') as hf:
            for batch_name in hf['data']:
                batch_group = hf['data'][batch_name]
                xc = batch_group['xc'][:]
                yc = batch_group['yc'][:]
                xt = batch_group['xt'][:]
                yt = batch_group['yt'][:]
                yield (xc, yc, xt, yt)

    # Define the output types for the generator
    output_signature = (
        tf.TensorSpec(shape=(None, None, None), dtype=dtype),
        tf.TensorSpec(shape=(None, None, None), dtype=dtype),
        tf.TensorSpec(shape=(None, None, None), dtype=dtype),
        tf.TensorSpec(shape=(None, None, None), dtype=dtype)
    )

    # Create a TensorFlow dataset from the generator
    data = tf.data.Dataset.from_generator(gen, output_signature=output_signature)
    metadata = hdf_get_metadata(filepath)
    return data, metadata


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
                if isinstance(item, (numpy.int64,numpy.uint8,numpy.float64)):
                    item = item.item()
                # Convert byte strings to Python strings
                elif isinstance(item, numpy.bytes_):
                    item = item.decode('utf-8')
                
                metadata_dict[name] = item
    return metadata_dict
