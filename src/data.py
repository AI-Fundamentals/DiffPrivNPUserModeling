"""
Implementations for the data generators used in the experiments.
Currently implemented in tensorflow. Ideally these would be implemented in lab?
"""

import h5py
import tensorflow as tf
import pdb


def print_hdf_structure(file_name):
    """
    Print the structure of an hdf5 file
    """
    with h5py.File(file_name, 'r') as f:
        f.visit(print)




# def load_hdf5_to_tensor(file_path):
#     """
#     Loads an HDF5 file into a TensorFlow tensor.
    
#     Parameters
#     ----------
#     file_path : str
#         The path to the HDF5 file.
    
#     Returns
#     -------
#     dict
#         A dictionary where each key is the name of a dataset in the HDF5 file and 
#         the corresponding value is a list of dereferenced data. If the data is a 
#         nested HDF5 reference, it is further dereferenced. All non-reference data 
#         is converted to a TensorFlow tensor before being appended to the list.
#     """
    
#     def dereference_data(dataset, f):
#         """
#         Dereferences a dataset of HDF5 references.
        
#         Parameters
#         ----------
#         dataset : h5py.Dataset
#             A dataset of HDF5 references.
#         f : h5py.File
#             The HDF5 file that contains the data to be dereferenced.
        
#         Returns
#         -------
#         list or tf.Tensor
#             If the dataset is a dataset of references, returns a list of dereferenced data. 
#             If the data is a nested HDF5 reference, it is further dereferenced. 
#             If the dataset is not a dataset of references, returns the dataset converted to a TensorFlow tensor.
#         """
#         if isinstance(dataset, h5py.Dataset) and isinstance(dataset[0], h5py.Reference):
#             data = []
#             for ref in dataset:
#                 dereferenced_data = f[h5py.h5r.get_name(ref, f.id)]
#                 if isinstance(dereferenced_data, h5py.Dataset) and isinstance(dereferenced_data[0], h5py.Reference):
#                     nested_data = dereference_data(dereferenced_data, f)
#                     data.append(nested_data)
#                 else:
#                     data.append(tf.convert_to_tensor(dereferenced_data))
#             return data
#         else:
#             return tf.convert_to_tensor(dataset)

#     # Open the HDF5 file
#     with h5py.File(file_path, 'r') as f:
#         data = {}
#         # Iterate over all datasets in the file
#         for dataset_name in f.keys():
#             # Convert the dataset name to a string if it's not already
#             if not isinstance(dataset_name, str):
#                 dataset_name = str(dataset_name)
#             # Get the dataset
#             dataset = f[dataset_name]

#             # Use the recursive function to dereference the data
#             data[dataset_name] = dereference_data(dataset, f)
#     return data




# %%

def hdf_to_tf_dataset(filepath, preload=True):
    """
    Load data from an HDF5 file and convert it into a TensorFlow dataset.
    Note that the output dataset will have a batch size of 1.

    Parameters
    ----------
    filepath : str
        The path to the HDF5 file.
    preload : bool, optional
        If True, all data will be loaded into memory at once. If False, a
        generator will be used to load the data at runtime. Setting it to
        True will be faster but you may run out of memory. Default is True.

    Returns
    -------
    data : tf.data.Dataset
        The TensorFlow dataset.

    """
    if preload:
        # Load all data into memory
        data = []
        with h5py.File(filepath, 'r') as hf:
            for sample_name in hf:
                sample_group = hf[sample_name]
                xc = sample_group['xc'][:]
                yc = sample_group['yc'][:]
                xt = sample_group['xt'][:]
                yt = sample_group['yt'][:]
                data.append((xc, yc, xt, yt))

        # Convert your data to a TensorFlow dataset
        data = tf.data.Dataset.from_tensor_slices(data)
    else:
        # Define a generator for the data
        def gen():
            with h5py.File(filepath, 'r') as hf:
                for sample_name in hf:
                    sample_group = hf[sample_name]
                    xc = sample_group['xc'][:]
                    yc = sample_group['yc'][:]
                    xt = sample_group['xt'][:]
                    yt = sample_group['yt'][:]
                    yield (xc, yc, xt, yt)

        # Define the output types for the generator
        output_signature = (
            tf.TensorSpec(shape=(None, None, None), dtype=tf.float32),
            tf.TensorSpec(shape=(None, None, None), dtype=tf.float32),
            tf.TensorSpec(shape=(None, None, None), dtype=tf.float32),
            tf.TensorSpec(shape=(None, None, None), dtype=tf.float32)
        )

        # Create a TensorFlow dataset from the generator
        data = tf.data.Dataset.from_generator(gen, output_signature=output_signature)

    return data
