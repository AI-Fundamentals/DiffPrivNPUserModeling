import h5py
import numpy as np
import tensorflow as tf

from dppum.data import hdf_get_metadata

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





def hdf_to_dataset_pad_tf(filepath, n_users=16, batch_size=1, padding_values=-1.,dtype=tf.float32):
    """
    Load data from an HDF5 file and convert it into a padded TensorFlow dataset.

    Parameters
    ----------
    filepath : str
        The path to the HDF5 file.
    n_users : int
        The number of users to load from the file.
    batch_size : int
        The batch size (users per batch) for the batched dataset. Default is 1.
    padding_values : float
        The value to use for padding in the output dataset.
    dtype : tf.DType, optional
        The data type for the output data. Default is tf.float32.

    Returns
    -------
    data : tf.data.Dataset
        The TensorFlow dataset.
    metadata : dict
        A dictionary containing metadata for the dataset. Some is loaded from
        the file, some is based on the arguments above.

    """
    # Define a generator to load the data from the HDF file
    def gen():
        with h5py.File(filepath, 'r') as hf:
            # Get the names of the batches
            batch_names = list(hf['data'].keys())
            # Shuffle the batch names
            random.shuffle(batch_names)
            # Select the first n_samples
            selected_batches = batch_names[:n_users]
            
            data = []
            for batch_name in selected_batches:
                batch_group = hf['data'][batch_name]
                xc = batch_group['xc'][:]
                yc = batch_group['yc'][:]
                xt = batch_group['xt'][:]
                yt = batch_group['yt'][:]
                data.append((xc, yc, xt, yt))
            for item in data:
                yield item

    # Define the output types for the generator
    output_signature = (
        tf.TensorSpec(shape=(None, None, None), dtype=dtype),
        tf.TensorSpec(shape=(None, None, None), dtype=dtype),
        tf.TensorSpec(shape=(None, None, None), dtype=dtype),
        tf.TensorSpec(shape=(None, None, None), dtype=dtype)
    )

    # Create a TensorFlow dataset from the generator
    data = tf.data.Dataset.from_generator(gen, output_signature=output_signature)
    # Batch the dataset
    data = data.padded_batch(batch_size,padding_values=padding_values)
    
    # Load metadata from the HDF file
    metadata = hdf_get_metadata(filepath)
    # Augment the metadata
    metadata['n_users'] = n_users
    metadata['batch_size'] = batch_size
    metadata['n_batches'] = int(np.ceil(n_users/batch_size))
    
    return data, metadata