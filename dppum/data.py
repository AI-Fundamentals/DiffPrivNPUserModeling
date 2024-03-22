"""
Implementations for the data generators used in the experiments.
Currently implemented in tensorflow. Ideally these would be implemented in lab?
"""

import h5py
import tensorflow as tf
import pdb
import numpy as np
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


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



# %%% Pytorch functions



class HDFDataset_torch(Dataset):
    """
    A PyTorch Dataset for reading data from an HDF5 file.

    Parameters
    ----------
    filepath : str
        The path to the HDF5 file.
    n_users : int, optional
        The number of users to select, by default 16

    Attributes
    ----------
    filepath : str
        The path to the HDF5 file.
    n_users : int
        The number of users to select from the file.
    batch_names : list
        The names of the batches in the HDF5 file.
    selected_batches : list
        The names of the selected batches.
    """

    def __init__(self, filepath, n_users=16):
        self.filepath = filepath
        self.n_users = n_users
        with h5py.File(self.filepath, 'r') as hf:
            self.batch_names = list(hf['data'].keys())
        np.random.shuffle(self.batch_names)
        self.selected_batches = self.batch_names[:self.n_users]

    def __len__(self):
        """
        Returns the number of selected batches.

        Returns
        -------
        int
            The number of selected batches.
        """
        return len(self.selected_batches)

    def __getitem__(self, idx):
        """
        Returns the data for a given batch index.

        Parameters
        ----------
        idx : int
            The index of the batch to return.

        Returns
        -------
        tuple
            The xc, yc, xt, yt data for the batch.
        """
        with h5py.File(self.filepath, 'r') as hf:
            batch_name = self.selected_batches[idx]
            batch_group = hf['data'][batch_name]
            xc = torch.from_numpy(batch_group['xc'][:])
            yc = torch.from_numpy(batch_group['yc'][:])
            xt = torch.from_numpy(batch_group['xt'][:])
            yt = torch.from_numpy(batch_group['yt'][:])
        return xc, yc, xt, yt




def hdf_to_dataset_pad_torch(filepath, n_users=16, batch_size=1, padding_values=-1.):
    """
    Load an HDF dataset and pad it into a PyTorch DataLoader.

    Parameters
    ----------
    filepath : str
        Path to the HDF file.
    n_users : int, optional
        Number of users to be loaded from the dataset. Default is 16.
    batch_size : int, optional
        Size of the batches. Default is 1.
    padding_values : float, optional
        Value used for padding, Default is -1.

    Returns
    -------
    torch.utils.data.DataLoader
        DataLoader object with the padded dataset.

    """
    dataset = HDFDataset_torch(filepath, n_users)
    
    def custom_pad(data, padding_values=-1.):
        """
        Custom padding function for 3D tensors that vary in the size of
        dimensions 0 and 2.

        Parameters
        ----------
        data : list of torch.Tensor
            List of tensors to be padded.
        padding_values : float, optional
            Value used for padding. Default is -1.

        Returns
        -------
        torch.Tensor
            Padded tensor.

        """
        max_len = max([x.size(0) for x in data])
        max_dim2 = max([x.size(2) for x in data])
        padded = torch.full((len(data), max_len, data[0].size(1), max_dim2), padding_values)
        for i, tensor in enumerate(data):
            padded[i, :tensor.size(0), :, :tensor.size(2)] = tensor
        return padded
    
    def collate_fn(batch):
        """
        Function to collate data into a padded batch.

        Parameters
        ----------
        batch : list
            List of data to be collated.

        Returns
        -------
        tuple of torch.Tensor
            Tuple of collated, padded data.

        """
        xc, yc, xt, yt = zip(*batch)
        xc = custom_pad(xc, padding_values=padding_values)
        yc = custom_pad(yc, padding_values=padding_values)
        xt = custom_pad(xt, padding_values=padding_values)
        yt = custom_pad(yt, padding_values=padding_values)
        return xc, yc, xt, yt
    
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)
    return dataloader


